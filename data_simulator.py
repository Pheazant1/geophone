"""
data_simulator.py
=================
Synthetic geophone array generator for the Seismic Perimeter Intelligence core.

A geophone is a passive velocity transducer: it outputs a small voltage that is
proportional to the vertical velocity of the ground surface. This module
produces realistic synthetic feeds for a four-element array, so that the full
conditioning, localisation and classification stack can be developed, tested and
demonstrated against a known ground truth, ahead of bringing the physical sensor
array online.

Array model
-----------
Four geophones sit at the corners of the protected area. A disturbance such as a
footfall or a wheel axle happens at some point on the ground and radiates a
surface wave outward in all directions. Each sensor therefore sees the same
event, but:

  * delayed, because the wave takes time to travel (the nearest sensor fires
    first and the farthest one last), and
  * attenuated, because surface waves spread out and the ground absorbs energy
    with distance.

Those per-sensor delays and amplitudes are exactly what the localisation stage
later inverts to recover where the event happened and which way it is moving.

Signal model
------------
Each ground impact is a damped sinusoid inside the seismic band of interest
(roughly 10 to 50 Hz). At a fixed reference distance the peak amplitude of an
impact is linear in the dynamic load that caused it:

        reference_amplitude  =  coupling_constant  x  effective_mass

That linear law, combined with the distance attenuation above, is what lets the
classifier recover a range-corrected mass from the measured amplitude.

Depends only on numpy. The default sampling rate is 100 Hz.

Run directly to generate an array feed and print a summary, or to export a CSV:

    python data_simulator.py
    python data_simulator.py --duration 30 --seed 7 --csv sample_feed.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Acquisition and site calibration
# ---------------------------------------------------------------------------

# Default acquisition rate of the data logger, in samples per second.
DEFAULT_SAMPLING_RATE_HZ: int = 100

# Background noise floor of a buried geophone plus its amplifier, as a standard
# deviation in volts. Each channel carries its own independent noise.
DEFAULT_NOISE_FLOOR_V: float = 0.0015

# Geometry of the array. Four elements at the corners of a 30 m square. The
# origin is one corner; coordinates are in metres.
SENSOR_POSITIONS_M: np.ndarray = np.array(
    [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]], dtype=float)

# Surface (Rayleigh) wave speed in the local soil, in metres per second. This
# is a site survey value; firm soils sit in the low hundreds.
WAVE_SPEED_M_S: float = 250.0

# Reference distance at which the coupling constants below are defined, in
# metres, plus the anelastic absorption coefficient per metre. Together with
# geometric spreading these set how amplitude falls off with range.
REF_DISTANCE_M: float = 5.0
ABSORPTION_PER_M: float = 0.01

# Physics-informed coupling constants in volts of geophone output per kilogram
# of dynamic load, referenced to REF_DISTANCE_M. They are linear in load and
# differ between a footfall and a rolling wheel because the two couple energy
# into the ground through very different contact mechanics.
FOOTSTEP_COUPLING_V_PER_KG: float = 8.0e-4
AXLE_COUPLING_V_PER_KG: float = 1.2e-4


# ---------------------------------------------------------------------------
# Ground-truth bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class InjectedEvent:
    """Record of one disturbance written into a feed, including its true path.

    The processing stack never sees this; it exists so tests and demos can
    compare their conclusions, including the recovered location and heading,
    against the truth.
    """

    kind: str                              # "human" or "vehicle"
    start_s: float
    end_s: float
    reference_peak_v: float                # peak at the reference distance
    mass_kg: float
    path_start_xy: Tuple[float, float]
    path_end_xy: Tuple[float, float]
    detail: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Array simulator
# ---------------------------------------------------------------------------

class GeophoneArraySimulator:
    """Builds synthetic multi-channel geophone feeds one disturbance at a time.

    Typical use:

        sim = GeophoneArraySimulator(seed=7)
        feed = sim.baseline(duration_s=30.0)            # shape (4, n)
        sim.add_footstep_train(feed, start_s=3.0,
                               path_start_xy=(10, 12),
                               path_end_xy=(16, 18), mass_kg=75.0)
    """

    def __init__(self, sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
                 seed: int | None = None,
                 sensor_positions: np.ndarray = SENSOR_POSITIONS_M,
                 wave_speed_m_s: float = WAVE_SPEED_M_S,
                 absorption_per_m: float = ABSORPTION_PER_M) -> None:
        self.fs: int = int(sampling_rate_hz)
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.sensor_positions = np.asarray(sensor_positions, dtype=float)
        self.wave_speed = float(wave_speed_m_s)
        self.absorption = float(absorption_per_m)
        self.events: List[InjectedEvent] = []

    @property
    def n_sensors(self) -> int:
        return self.sensor_positions.shape[0]

    # -- baseline ----------------------------------------------------------

    def baseline(self, duration_s: float,
                 noise_floor_v: float = DEFAULT_NOISE_FLOOR_V) -> np.ndarray:
        """Return a fresh multi-channel feed of just the noise floor.

        Each channel gets independent white Gaussian noise plus a slow
        sinusoidal wander standing in for wind loading and thermal drift. The
        independence across channels is what keeps the noise from correlating
        and confusing the localiser. Calling this resets the event log.
        """
        n = int(round(duration_s * self.fs))
        t = np.arange(n) / self.fs
        feed = np.empty((self.n_sensors, n), dtype=float)
        for i in range(self.n_sensors):
            white = self.rng.normal(0.0, noise_floor_v, n)
            wander = 0.3 * noise_floor_v * np.sin(
                2 * np.pi * 0.05 * t + self.rng.uniform(0.0, 2.0 * np.pi))
            feed[i] = white + wander
        self.events = []
        return feed

    # -- propagation -------------------------------------------------------

    def _propagation(self, source_xy: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-sensor (travel delay in seconds, amplitude factor).

        The amplitude factor combines geometric spreading of a surface wave
        (one over the square root of range) with anelastic absorption (an
        exponential decay in range), normalised to unity at the reference
        distance.
        """
        deltas = self.sensor_positions - np.asarray(source_xy, dtype=float)
        distances = np.linalg.norm(deltas, axis=1)
        delays = distances / self.wave_speed
        d_eff = np.maximum(distances, REF_DISTANCE_M)
        amp = np.sqrt(REF_DISTANCE_M / d_eff) * np.exp(
            -self.absorption * (d_eff - REF_DISTANCE_M))
        return delays, amp

    # -- low level pulse shaping ------------------------------------------

    def _damped_pulse(self, freq_hz: float, decay_tau_s: float,
                      attack_s: float = 0.005,
                      duration_s: float | None = None) -> np.ndarray:
        """Create one unit-amplitude damped sinusoid (a single ground impact)."""
        if duration_s is None:
            duration_s = max(6.0 * decay_tau_s, 0.2)
        n = int(round(duration_s * self.fs))
        t = np.arange(n) / self.fs
        envelope = np.exp(-t / decay_tau_s)
        attack = np.ones(n)
        a = int(round(attack_s * self.fs))
        if a > 1:
            ramp = np.arange(a) / a
            attack[:a] = 0.5 * (1.0 - np.cos(np.pi * ramp))
        carrier = np.sin(2.0 * np.pi * freq_hz * t)
        return attack * envelope * carrier

    def _add_at(self, channel: np.ndarray, pulse: np.ndarray,
                start_s: float) -> None:
        """Add a pulse into one channel in place, starting at start_s."""
        start = int(round(start_s * self.fs))
        if start < 0 or start >= channel.size:
            return
        end = min(start + pulse.size, channel.size)
        channel[start:end] += pulse[: end - start]

    def _emit(self, feed: np.ndarray, source_xy: Sequence[float],
              emit_time_s: float, reference_amp_v: float, freq_hz: float,
              decay_tau_s: float, attack_s: float) -> None:
        """Emit one impact from a point on the ground into every channel.

        Each channel receives the same pulse shape, delayed by the travel time
        to that sensor and scaled by the distance attenuation to it.
        """
        delays, amp = self._propagation(source_xy)
        shape = self._damped_pulse(freq_hz, decay_tau_s, attack_s)
        for i in range(self.n_sensors):
            self._add_at(feed[i], reference_amp_v * amp[i] * shape,
                         emit_time_s + delays[i])

    def _emit_rumble(self, feed: np.ndarray, source_xy: Sequence[float],
                     start_s: float, end_s: float, fundamental_hz: float,
                     harmonics: Sequence[int], amp_v: float) -> None:
        """Emit continuous engine rumble from a point into every channel."""
        delays, amp = self._propagation(source_xy)
        start_idx = int(round(start_s * self.fs))
        n = int(round((end_s - start_s) * self.fs))
        if n <= 1:
            return
        t = np.arange(n) / self.fs
        window = np.hanning(n)
        base = np.zeros(n)
        for h in harmonics:
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            base += (amp_v / h) * np.sin(2.0 * np.pi * fundamental_hz * h * t + phase)
        base *= window
        for i in range(self.n_sensors):
            self._add_at(feed[i], amp[i] * base,
                         start_s + delays[i])

    # -- footsteps ---------------------------------------------------------

    def add_footstep_train(self, feed: np.ndarray, start_s: float,
                           path_start_xy: Sequence[float],
                           path_end_xy: Sequence[float],
                           num_steps: int = 8, cadence_hz: float = 1.8,
                           mass_kg: float = 75.0, decay_tau_s: float = 0.06,
                           freq_hz: float = 30.0, timing_jitter: float = 0.03,
                           asymmetry: float = 0.05, surface_fn=None,
                           label: str = "footsteps") -> np.ndarray:
        """Inject a walking human moving along a straight path.

        Each footfall is a sharp heel-strike pulse. Steps repeat at
        ``cadence_hz`` and the walker advances from ``path_start_xy`` to
        ``path_end_xy`` across the train, so successive footfalls are emitted
        from successive points on the ground. Left and right feet differ
        slightly in amplitude and timing, forming the gait signature. The
        reference amplitude of each footfall scales linearly with ``mass_kg``.
        """
        path_start = np.asarray(path_start_xy, dtype=float)
        path_end = np.asarray(path_end_xy, dtype=float)
        step_interval = 1.0 / cadence_hz
        reference_amp = FOOTSTEP_COUPLING_V_PER_KG * mass_kg
        cursor = start_s
        for i in range(num_steps):
            frac = i / max(num_steps - 1, 1)
            position = path_start + (path_end - path_start) * frac
            f_mult, a_mult = surface_fn(position) if surface_fn else (1.0, 1.0)
            foot_gain = 1.0 + (asymmetry if i % 2 else -asymmetry)
            amp = reference_amp * foot_gain * a_mult * (1.0 + self.rng.normal(0.0, 0.02))
            freq = freq_hz * f_mult * (1.0 + self.rng.normal(0.0, 0.03))
            self._emit(feed, position, cursor, amp, freq, decay_tau_s, 0.005)
            cursor += step_interval + self.rng.normal(0.0, timing_jitter * step_interval)

        self.events.append(InjectedEvent(
            kind="human", start_s=start_s, end_s=cursor,
            reference_peak_v=reference_amp, mass_kg=mass_kg,
            path_start_xy=tuple(path_start), path_end_xy=tuple(path_end),
            detail={"cadence_hz": cadence_hz, "num_steps": num_steps,
                    "label": label}))
        return feed

    # -- vehicles ----------------------------------------------------------

    def add_vehicle_pass(self, feed: np.ndarray, start_s: float,
                         path_start_xy: Sequence[float],
                         path_end_xy: Sequence[float],
                         axle_masses: Sequence[float] = (650.0, 650.0),
                         axle_spacing_s: float = 0.45,
                         axle_freq_hz: float = 14.0,
                         axle_decay_tau_s: float = 0.16,
                         rumble_fundamental_hz: float = 9.0,
                         rumble_harmonics: Sequence[int] = (1, 2, 3),
                         rumble_amp_v: float = 0.006, surface_fn=None,
                         label: str = "vehicle") -> np.ndarray:
        """Inject a vehicle driving along a path past the array.

        Axles present to the array at evenly spaced moments, so each axle
        deflection lands ``axle_spacing_s`` after the previous one, emitted from
        the next point along the path. Axle deflections are lower in frequency
        and broader than footfalls, and each axle's reference amplitude scales
        linearly with the load on that axle. A continuous engine rumble runs
        across the whole pass.
        """
        path_start = np.asarray(path_start_xy, dtype=float)
        path_end = np.asarray(path_end_xy, dtype=float)
        n_axles = len(axle_masses)
        axle_times: List[float] = []
        cursor = start_s
        for j, mass in enumerate(axle_masses):
            frac = j / max(n_axles - 1, 1)
            position = path_start + (path_end - path_start) * frac
            f_mult, a_mult = surface_fn(position) if surface_fn else (1.0, 1.0)
            amp = AXLE_COUPLING_V_PER_KG * mass * a_mult * (1.0 + self.rng.normal(0.0, 0.02))
            freq = axle_freq_hz * f_mult * (1.0 + self.rng.normal(0.0, 0.03))
            self._emit(feed, position, cursor, amp, freq, axle_decay_tau_s, 0.01)
            axle_times.append(cursor)
            cursor += axle_spacing_s

        mid = 0.5 * (path_start + path_end)
        self._emit_rumble(feed, mid, axle_times[0] - 0.3, axle_times[-1] + 0.6,
                          rumble_fundamental_hz, rumble_harmonics, rumble_amp_v)

        self.events.append(InjectedEvent(
            kind="vehicle", start_s=start_s, end_s=axle_times[-1] + 0.6,
            reference_peak_v=AXLE_COUPLING_V_PER_KG * max(axle_masses),
            mass_kg=float(sum(axle_masses)),
            path_start_xy=tuple(path_start), path_end_xy=tuple(path_end),
            detail={"axle_masses": list(axle_masses),
                    "axle_spacing_s": axle_spacing_s,
                    "num_axles": n_axles, "label": label}))
        return feed


# ---------------------------------------------------------------------------
# Reference scenario
# ---------------------------------------------------------------------------

def build_demo_scenario(fs: int = DEFAULT_SAMPLING_RATE_HZ,
                        duration_s: float = 30.0,
                        seed: int | None = 7) -> np.ndarray:
    """Assemble the canonical demonstration timeline used across the project.

    The 30 second, four-channel feed contains, in order:

      t =  3 s  property owner on foot, walking north-east     -> known
      t = 10 s  unidentified person on foot, walking north-west -> unknown
      t = 16 s  owner vehicle at the gate, unladen              -> known
      t = 23 s  owner vehicle at the same gate, rear axle loaded -> anomaly

    The two vehicle passes use the same gate location, so their geometry is
    identical and the only difference the system should report is the added
    rear-axle load. The known events are recognised and suppressed; the unknown
    walker and the overloaded vehicle each raise an alert downstream.
    """
    sim = GeophoneArraySimulator(sampling_rate_hz=fs, seed=seed)
    feed = sim.baseline(duration_s=duration_s)

    # Recognised owner walking the perimeter toward the north-east.
    sim.add_footstep_train(feed, start_s=3.0, path_start_xy=(10.0, 12.0),
                           path_end_xy=(16.0, 18.0), num_steps=8,
                           cadence_hz=1.8, mass_kg=75.0, label="owner_on_foot")

    # An unfamiliar, heavier person moving faster toward the north-west.
    sim.add_footstep_train(feed, start_s=10.0, path_start_xy=(22.0, 9.0),
                           path_end_xy=(17.0, 15.0), num_steps=8,
                           cadence_hz=2.4, mass_kg=95.0, freq_hz=32.0,
                           label="unknown_on_foot")

    # The owner's registered vehicle at the gate, empty.
    sim.add_vehicle_pass(feed, start_s=16.0, path_start_xy=(15.0, 22.0),
                         path_end_xy=(15.0, 22.0), axle_masses=(650.0, 650.0),
                         label="owner_vehicle")

    # The same vehicle at the same gate with a 20 percent heavier rear axle.
    sim.add_vehicle_pass(feed, start_s=23.0, path_start_xy=(15.0, 22.0),
                         path_end_xy=(15.0, 22.0), axle_masses=(650.0, 780.0),
                         label="owner_vehicle_loaded")

    return feed


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------

def _summarise(feed: np.ndarray, fs: int) -> None:
    duration = feed.shape[1] / fs
    print("Synthetic geophone array feed")
    print("  sampling rate : {0} Hz".format(fs))
    print("  channels      : {0}".format(feed.shape[0]))
    print("  duration      : {0:.1f} s ({1} samples/channel)".format(
        duration, feed.shape[1]))
    for i in range(feed.shape[0]):
        print("  channel {0}     : peak {1:.4f} V, rms {2:.4f} V".format(
            i, float(np.max(np.abs(feed[i]))),
            float(np.sqrt(np.mean(feed[i] ** 2)))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic four-channel geophone feed.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="feed length in seconds (default 30)")
    parser.add_argument("--fs", type=int, default=DEFAULT_SAMPLING_RATE_HZ,
                        help="sampling rate in Hz (default 100)")
    parser.add_argument("--seed", type=int, default=7,
                        help="random seed for reproducibility (default 7)")
    parser.add_argument("--csv", type=str, default=None,
                        help="optional path to write time and per-channel volts")
    args = parser.parse_args()

    feed = build_demo_scenario(fs=args.fs, duration_s=args.duration,
                               seed=args.seed)
    _summarise(feed, args.fs)

    if args.csv:
        t = np.arange(feed.shape[1]) / args.fs
        table = np.column_stack([t, feed.T])
        header = "time_s," + ",".join(
            "ch{0}_v".format(i) for i in range(feed.shape[0]))
        np.savetxt(args.csv, table, delimiter=",", header=header,
                   comments="", fmt="%.6f")
        print("  wrote         : {0}".format(args.csv))


if __name__ == "__main__":
    main()

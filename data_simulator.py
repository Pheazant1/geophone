"""
data_simulator.py
==================
Synthetic geophone signal generator for the Seismic Perimeter Intelligence core.

A geophone is a passive velocity transducer: it outputs a small voltage that is
proportional to the vertical velocity of the ground surface. This module
produces realistic synthetic voltage feeds so that the full conditioning and
classification stack can be developed, tested and demonstrated against a known
ground truth, ahead of bringing the physical sensor array online.

Signal model
------------
Every ground disturbance (a footfall, or a wheel axle passing the sensor) is
modelled as a damped sinusoid living inside the seismic band of interest
(roughly 10 to 50 Hz). The key physical assumption is that the peak amplitude
of each disturbance is, to first order, linear in the dynamic load that caused
it:

        voltage_amplitude  =  coupling_constant  x  effective_mass

That single linear relationship is what later allows the classifier to recover
an effective mass from a measured amplitude, and therefore to notice when a
recognised vehicle returns carrying extra payload.

The module depends only on numpy. The default sampling rate is 100 Hz.

Run directly to generate a feed and print a summary, or to export a CSV:

    python data_simulator.py
    python data_simulator.py --duration 30 --seed 7 --csv sample_feed.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

# Default acquisition rate of the data logger, in samples per second.
DEFAULT_SAMPLING_RATE_HZ: int = 100

# Background noise floor of a buried geophone plus its amplifier, expressed as
# a standard deviation in volts. Real installations sit in the low millivolts.
DEFAULT_NOISE_FLOOR_V: float = 0.0015

# Physics-informed coupling constants in volts of geophone output per kilogram
# of dynamic load, referenced to a nominal coupling distance. They are
# deliberately linear: double the load and the peak amplitude doubles. The two
# values differ because a footfall and a rolling wheel couple energy into the
# ground through very different contact mechanics.
FOOTSTEP_COUPLING_V_PER_KG: float = 8.0e-4
AXLE_COUPLING_V_PER_KG: float = 1.2e-4


# ---------------------------------------------------------------------------
# Ground-truth bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class InjectedEvent:
    """Description of one disturbance that was written into a feed.

    This is purely a record of what the simulator placed into the signal. The
    processing and classification stack never sees it; it exists so tests and
    demos can compare their conclusions against the truth.
    """

    kind: str                       # "human" or "vehicle"
    start_s: float                  # onset time in seconds
    end_s: float                    # approximate end time in seconds
    peak_v: float                   # nominal peak amplitude in volts
    mass_kg: float                  # effective mass that produced the event
    detail: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class GeophoneSimulator:
    """Builds synthetic geophone voltage feeds one disturbance at a time.

    Typical use:

        sim = GeophoneSimulator(seed=7)
        feed = sim.baseline(duration_s=30.0)
        sim.add_footstep_train(feed, start_s=3.0, mass_kg=75.0)
        sim.add_vehicle_pass(feed, start_s=16.0, axle_masses=[650.0, 650.0])
    """

    def __init__(self, sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
                 seed: int | None = None) -> None:
        self.fs: int = int(sampling_rate_hz)
        self.rng: np.random.Generator = np.random.default_rng(seed)
        # Every disturbance added to the current feed is logged here.
        self.events: List[InjectedEvent] = []

    # -- baseline ----------------------------------------------------------

    def baseline(self, duration_s: float,
                 noise_floor_v: float = DEFAULT_NOISE_FLOOR_V) -> np.ndarray:
        """Return a fresh feed containing only the environmental noise floor.

        The floor is white Gaussian electronic noise plus a very slow sinusoidal
        wander that stands in for wind loading and thermal drift. Calling this
        method resets the injected-event log.
        """
        n = int(round(duration_s * self.fs))
        t = np.arange(n) / self.fs
        white = self.rng.normal(0.0, noise_floor_v, n)
        # Slow wander (0.05 Hz) that the adaptive baseline downstream must learn
        # to ignore rather than alarm on.
        wander = 0.3 * noise_floor_v * np.sin(
            2 * np.pi * 0.05 * t + self.rng.uniform(0.0, 2.0 * np.pi))
        self.events = []
        return white + wander

    # -- low level pulse shaping ------------------------------------------

    def _damped_pulse(self, amplitude: float, freq_hz: float,
                      decay_tau_s: float, attack_s: float = 0.005,
                      duration_s: float | None = None) -> np.ndarray:
        """Create one damped sinusoid (a single ground impact).

        The shape is a sine carrier inside a sharp rising edge followed by an
        exponential decay. A short heel strike uses a small ``decay_tau_s``; a
        broad axle deflection uses a larger one.
        """
        if duration_s is None:
            duration_s = max(6.0 * decay_tau_s, 0.2)
        n = int(round(duration_s * self.fs))
        t = np.arange(n) / self.fs

        envelope = np.exp(-t / decay_tau_s)

        # Raised-cosine attack so the onset is fast but not a numerical step.
        attack = np.ones(n)
        a = int(round(attack_s * self.fs))
        if a > 1:
            ramp = np.arange(a) / a
            attack[:a] = 0.5 * (1.0 - np.cos(np.pi * ramp))

        carrier = np.sin(2.0 * np.pi * freq_hz * t)
        return amplitude * attack * envelope * carrier

    def _add_at(self, feed: np.ndarray, pulse: np.ndarray,
                start_s: float) -> None:
        """Add a pulse into ``feed`` in place, starting at ``start_s``."""
        start = int(round(start_s * self.fs))
        if start < 0 or start >= feed.size:
            return
        end = min(start + pulse.size, feed.size)
        feed[start:end] += pulse[: end - start]

    def _add_rumble(self, feed: np.ndarray, start_s: float, end_s: float,
                    fundamental_hz: float, harmonics: Sequence[int],
                    amp_v: float) -> None:
        """Add continuous engine rumble across a vehicle pass window.

        The rumble is a sum of harmonics of a low fundamental, shaped by a Hann
        window so it rises and falls smoothly as the vehicle approaches and
        departs.
        """
        start = max(int(round(start_s * self.fs)), 0)
        end = min(int(round(end_s * self.fs)), feed.size)
        if end <= start:
            return
        n = end - start
        t = np.arange(n) / self.fs
        window = np.hanning(n) if n > 1 else np.ones(n)
        rumble = np.zeros(n)
        for h in harmonics:
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            # Higher harmonics carry less energy.
            rumble += (amp_v / h) * np.sin(2.0 * np.pi * fundamental_hz * h * t + phase)
        feed[start:end] += window * rumble

    # -- footsteps ---------------------------------------------------------

    def add_footstep_train(self, feed: np.ndarray, start_s: float,
                           num_steps: int = 8, cadence_hz: float = 1.8,
                           mass_kg: float = 75.0, decay_tau_s: float = 0.06,
                           freq_hz: float = 30.0, timing_jitter: float = 0.03,
                           asymmetry: float = 0.05,
                           label: str = "footsteps") -> np.ndarray:
        """Inject a walking human as a train of heel-strike impacts.

        Each footfall is a sharp spike with rapid decay. Steps repeat at
        ``cadence_hz`` (steps per second). Left and right feet differ slightly
        in amplitude (``asymmetry``) and arrive with small biological timing
        variation (``timing_jitter``), which together form the gait signature
        the classifier keys on. Peak amplitude scales linearly with ``mass_kg``.
        """
        step_interval = 1.0 / cadence_hz
        peak_amp = FOOTSTEP_COUPLING_V_PER_KG * mass_kg
        cursor = start_s
        for i in range(num_steps):
            # Alternate feet, then add small per-step amplitude variation.
            foot_gain = 1.0 + (asymmetry if i % 2 else -asymmetry)
            amp = peak_amp * foot_gain * (1.0 + self.rng.normal(0.0, 0.02))
            freq = freq_hz * (1.0 + self.rng.normal(0.0, 0.03))
            self._add_at(feed, self._damped_pulse(amp, freq, decay_tau_s), cursor)
            cursor += step_interval + self.rng.normal(0.0, timing_jitter * step_interval)

        self.events.append(InjectedEvent(
            kind="human", start_s=start_s, end_s=cursor, peak_v=peak_amp,
            mass_kg=mass_kg,
            detail={"cadence_hz": cadence_hz, "num_steps": num_steps,
                    "label": label}))
        return feed

    def add_overloaded_footstep_train(self, feed: np.ndarray, start_s: float,
                                      num_steps: int = 8,
                                      base_mass_kg: float = 75.0,
                                      payload_kg: float = 25.0,
                                      label: str = "overloaded_footsteps"
                                      ) -> np.ndarray:
        """Inject a person carrying a heavy load.

        Relative to an unburdened walk the peak amplitude rises (more mass), the
        impact decays more slowly (a heavier, flatter footfall) and the stance
        timing slows and grows more uneven. This is the on-foot analogue of a
        loaded vehicle.
        """
        total_mass = base_mass_kg + payload_kg
        return self.add_footstep_train(
            feed, start_s, num_steps=num_steps, cadence_hz=1.5,
            mass_kg=total_mass, decay_tau_s=0.09, freq_hz=28.0,
            timing_jitter=0.06, asymmetry=0.09, label=label)

    # -- vehicles ----------------------------------------------------------

    def add_vehicle_pass(self, feed: np.ndarray, start_s: float,
                         axle_masses: Sequence[float] = (650.0, 650.0),
                         axle_spacing_s: float = 0.45,
                         axle_freq_hz: float = 14.0,
                         axle_decay_tau_s: float = 0.16,
                         rumble_fundamental_hz: float = 9.0,
                         rumble_harmonics: Sequence[int] = (1, 2, 3),
                         rumble_amp_v: float = 0.006,
                         label: str = "vehicle") -> np.ndarray:
        """Inject a vehicle driving past the sensor.

        A vehicle at roughly constant speed presents its axles to the sensor at
        evenly spaced moments, so each axle deflection lands ``axle_spacing_s``
        after the previous one. Axle deflections are lower in frequency and
        broader than footfalls. Each axle's peak scales linearly with the load
        on that axle, so ``axle_masses`` controls the per-axle amplitudes and a
        heavier rear axle produces a taller rear peak. A continuous engine
        rumble (a low fundamental plus harmonics) runs across the whole pass.
        """
        axle_times: List[float] = []
        cursor = start_s
        for mass in axle_masses:
            amp = AXLE_COUPLING_V_PER_KG * mass * (1.0 + self.rng.normal(0.0, 0.02))
            freq = axle_freq_hz * (1.0 + self.rng.normal(0.0, 0.03))
            self._add_at(feed, self._damped_pulse(
                amp, freq, axle_decay_tau_s, attack_s=0.01), cursor)
            axle_times.append(cursor)
            cursor += axle_spacing_s

        pass_start = axle_times[0] - 0.3
        pass_end = axle_times[-1] + 0.6
        self._add_rumble(feed, pass_start, pass_end, rumble_fundamental_hz,
                         rumble_harmonics, rumble_amp_v)

        self.events.append(InjectedEvent(
            kind="vehicle", start_s=start_s, end_s=pass_end,
            peak_v=AXLE_COUPLING_V_PER_KG * max(axle_masses),
            mass_kg=float(sum(axle_masses)),
            detail={"axle_masses": list(axle_masses),
                    "axle_spacing_s": axle_spacing_s,
                    "num_axles": len(axle_masses), "label": label}))
        return feed


# ---------------------------------------------------------------------------
# Reference scenario
# ---------------------------------------------------------------------------

def build_demo_scenario(fs: int = DEFAULT_SAMPLING_RATE_HZ,
                        duration_s: float = 30.0,
                        seed: int | None = 7) -> np.ndarray:
    """Assemble the canonical demonstration timeline used across the project.

    The 30 second feed contains, in order:

      t =  3 s   property owner on foot      (75 kg, cadence 1.8 Hz)  -> known
      t = 10 s   unidentified person on foot (95 kg, cadence 2.4 Hz)  -> unknown
      t = 16 s   owner vehicle, unloaded     (axles 650 / 650 kg)     -> known
      t = 23 s   owner vehicle, rear loaded  (axles 650 / 780 kg)     -> anomaly

    The two known events should be recognised and suppressed; the unknown walker
    and the overloaded vehicle should each raise an alert downstream.
    """
    sim = GeophoneSimulator(sampling_rate_hz=fs, seed=seed)
    feed = sim.baseline(duration_s=duration_s)

    # Recognised owner walking the perimeter.
    sim.add_footstep_train(feed, start_s=3.0, num_steps=8, cadence_hz=1.8,
                           mass_kg=75.0, label="owner_on_foot")

    # An unfamiliar, heavier person moving faster.
    sim.add_footstep_train(feed, start_s=10.0, num_steps=8, cadence_hz=2.4,
                           mass_kg=95.0, freq_hz=32.0, label="unknown_on_foot")

    # The owner's registered vehicle, empty.
    sim.add_vehicle_pass(feed, start_s=16.0, axle_masses=(650.0, 650.0),
                         label="owner_vehicle")

    # The same vehicle returning with a 20 percent heavier rear axle.
    sim.add_vehicle_pass(feed, start_s=23.0, axle_masses=(650.0, 780.0),
                         label="owner_vehicle_loaded")

    return feed


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------

def _summarise(feed: np.ndarray, fs: int) -> None:
    duration = feed.size / fs
    print("Synthetic geophone feed")
    print("  sampling rate : {0} Hz".format(fs))
    print("  duration      : {0:.1f} s ({1} samples)".format(duration, feed.size))
    print("  peak amplitude: {0:.4f} V".format(float(np.max(np.abs(feed)))))
    print("  rms amplitude : {0:.4f} V".format(float(np.sqrt(np.mean(feed ** 2)))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic geophone voltage feed.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="feed length in seconds (default 30)")
    parser.add_argument("--fs", type=int, default=DEFAULT_SAMPLING_RATE_HZ,
                        help="sampling rate in Hz (default 100)")
    parser.add_argument("--seed", type=int, default=7,
                        help="random seed for reproducibility (default 7)")
    parser.add_argument("--csv", type=str, default=None,
                        help="optional path to write the feed as time,voltage")
    args = parser.parse_args()

    feed = build_demo_scenario(fs=args.fs, duration_s=args.duration,
                               seed=args.seed)
    _summarise(feed, args.fs)

    if args.csv:
        t = np.arange(feed.size) / args.fs
        table = np.column_stack([t, feed])
        np.savetxt(args.csv, table, delimiter=",", header="time_s,voltage_v",
                   comments="", fmt="%.6f")
        print("  wrote         : {0}".format(args.csv))


if __name__ == "__main__":
    main()

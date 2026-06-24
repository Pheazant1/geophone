"""
v1_demo.py
==========
End-to-end demonstration of the open-source V1 core: a self-learning seismic
pattern memory that starts from a blank slate.

V1 is deliberately not a security product. It carries no built-in idea of what a
human, a vehicle or an animal is. Instead it learns the cast of characters at a
site from scratch, the way a spam filter or a music-identification app does: it
listens for a while, builds up signatures of whatever keeps causing consistent
ground vibration, and from then on can tell a familiar signature from a new one
in a few seconds. That makes it a general tool for researchers, conservationists
and hobbyists who want to study or monitor whatever moves across their own
ground, without any labelled training data.

The run has two phases against the same growing memory:

  1. LEARNING. A stream of events arrives. The memory has no profiles at the
     start. Recurring signatures accrete observations and graduate to enrolled
     profiles; one-off oddities stay tentative and are treated as noise.

  2. GUARDING. New events arrive. Anything matching an enrolled signature is
     recognised and suppressed. Anything that matches nothing is flagged as a
     novel signature, which is the only thing the operator needs to look at.

The pipeline reuses the existing physics stack unchanged: the array simulator
for the feed, the signal processor for conditioning and event detection, and
the localiser for position. Only the decision layer is different. Where V2 asks
"what is this and is it allowed", V1 asks only "have I seen this before". The
learning itself lives in ``pattern_memory.py``.

Run it directly:

    python v1_demo.py
    python v1_demo.py --seed 11 --match-distance 1.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from data_simulator import (ABSORPTION_PER_M, DEFAULT_SAMPLING_RATE_HZ,
                            REF_DISTANCE_M, SENSOR_POSITIONS_M,
                            GeophoneArraySimulator)
from localizer import track_window
from pattern_memory import PatternMemory
from profile_classifier import FEATURE_PAD_S, extract_features
from signal_processor import SignalProcessor


# ---------------------------------------------------------------------------
# Label-free feature vector
# ---------------------------------------------------------------------------
#
# The fingerprint that identifies a source. Every entry is a physical quantity
# that does not depend on labels, and amplitude is range-corrected so the same
# source reads the same wherever it crosses the array. The companion scales are
# the natural repeatability of each feature between two sightings of one source;
# they are what makes the distance in pattern_memory interpretable.

FEATURE_NAMES: List[str] = [
    "log_ref_amp",        # log10 of range-corrected peak amplitude (source size)
    "dominant_freq_hz",   # spectral centre of the event (contact texture)
    "low_freq_ratio",     # fraction of band energy below the split (heavy vs sharp)
    "cadence_hz",         # impact-repetition rate (gait or axle rhythm)
]

FEATURE_SCALES: List[float] = [
    0.15,                 # log10 amplitude: mass jitter plus localisation range residual
    2.0,                  # dominant frequency, Hz
    0.06,                 # low-frequency energy ratio
    0.20,                 # cadence, Hz
]


def _cadence_hz(envelope_seg: np.ndarray, fs: int,
                fmin: float = 0.8, fmax: float = 4.0) -> float:
    """Estimate the impact-repetition rate from the envelope modulation spectrum.

    Peak-picking the envelope is fragile, and a plain autocorrelation is fooled
    by the slow rise and fall of amplitude as a source passes its closest point
    to a sensor, which can dominate the true step ripple. So the slow trend is
    first removed with a moving-average high-pass, leaving the periodic gait or
    axle ripple, and the dominant frequency of that ripple is read off a
    zero-padded spectrum restricted to plausible cadences. This is robust to a
    missed step and to left-right amplitude asymmetry, and it cleanly separates
    walkers of different pace. Returns 0 when no periodic structure is present.
    """
    n = envelope_seg.size
    if n < int(fs * 0.8) or not np.any(np.abs(envelope_seg) > 0):
        return 0.0

    # High-pass: subtract the slow amplitude trend of the source passing by,
    # keeping the step-to-step ripple.
    trend_win = max(3, int(round(1.2 * fs)))
    trend = uniform_filter1d(envelope_seg, size=trend_win, mode="nearest")
    ripple = envelope_seg - trend
    if not np.any(np.abs(ripple) > 0):
        return 0.0

    # Zero-pad the windowed ripple for fine frequency resolution, then take the
    # strongest line inside the cadence band.
    nfft = 1024
    spectrum = np.abs(np.fft.rfft(ripple * np.hanning(n), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any() or spectrum[band].sum() == 0.0:
        return 0.0
    band_freqs = freqs[band]
    return float(band_freqs[int(np.argmax(spectrum[band]))])


def _amplitude_factor(distance_m: float) -> float:
    """Surface-wave amplitude factor at a range, normalised to the reference.

    Mirrors the propagation model in the simulator: geometric spreading as one
    over the square root of range, times anelastic absorption. Dividing a
    measured peak by this recovers the amplitude the source would have produced
    at the calibration distance, making the amplitude feature range-independent.
    """
    d = max(distance_m, REF_DISTANCE_M)
    return float(np.sqrt(REF_DISTANCE_M / d) * np.exp(-ABSORPTION_PER_M * (d - REF_DISTANCE_M)))


def build_feature_vector(result, window, fs: int) -> Tuple[np.ndarray, float, dict]:
    """Turn one detected activity window into a label-free signature vector.

    Returns the feature vector, the source range in metres, and a small dict of
    the human-readable feature values for logging.
    """
    features = extract_features(result, window, fs)
    track = track_window(result.filtered, window.start_idx, window.end_idx, fs)

    sensor = SENSOR_POSITIONS_M[window.best_channel]
    distance = float(np.hypot(sensor[0] - track.position.x_m,
                              sensor[1] - track.position.y_m))
    reference_amp = features.peak_v / _amplitude_factor(distance)
    log_ref_amp = float(np.log10(max(reference_amp, 1e-9)))

    # Robust gait or axle rhythm from the strongest channel's envelope.
    pad = int(round(FEATURE_PAD_S * fs))
    s = max(0, window.start_idx - pad)
    e = min(result.envelope.shape[1], window.end_idx + pad)
    cadence_hz = _cadence_hz(result.envelope[window.best_channel][s:e], fs)

    vector = np.array([
        log_ref_amp,
        features.dominant_freq_hz,
        features.low_freq_ratio,
        cadence_hz,
    ], dtype=float)

    readable = {
        "ref_amp_v": round(reference_amp, 4),
        "dominant_freq_hz": round(features.dominant_freq_hz, 1),
        "low_freq_ratio": round(features.low_freq_ratio, 3),
        "cadence_hz": round(cadence_hz, 2),
        "range_m": round(distance, 1),
    }
    return vector, distance, readable


# ---------------------------------------------------------------------------
# Anonymous entities for the simulated site
# ---------------------------------------------------------------------------
#
# Ground truth used only to build the feed and, afterwards, to check that the
# memory clustered correctly. The memory itself never sees these keys.

def _emit_entity(sim: GeophoneArraySimulator, feed: np.ndarray, key: str,
                 t_s: float, rng: np.random.Generator) -> None:
    """Inject one appearance of a known entity, with natural variation.

    Each call jitters mass, cadence and path so repeat visits of the same entity
    are similar but never identical, which is what the matcher has to cope with.
    """
    if key == "A":          # a light, unhurried walker
        start = np.array([8.0, 10.0]) + rng.normal(0.0, 1.5, 2)
        end = start + np.array([6.0, 6.0]) + rng.normal(0.0, 1.0, 2)
        sim.add_footstep_train(feed, start_s=t_s, path_start_xy=start,
                               path_end_xy=end, num_steps=6,
                               cadence_hz=float(rng.normal(1.7, 0.05)),
                               mass_kg=float(rng.normal(70.0, 3.0)),
                               freq_hz=30.0, label="A")
    elif key == "B":        # a heavier, quicker walker
        start = np.array([22.0, 8.0]) + rng.normal(0.0, 1.5, 2)
        end = start + np.array([-5.0, 6.0]) + rng.normal(0.0, 1.0, 2)
        sim.add_footstep_train(feed, start_s=t_s, path_start_xy=start,
                               path_end_xy=end, num_steps=6,
                               cadence_hz=float(rng.normal(2.3, 0.06)),
                               mass_kg=float(rng.normal(95.0, 3.0)),
                               freq_hz=32.0, label="B")
    elif key == "C":        # a recurring vehicle at the gate
        gate = np.array([15.0, 22.0]) + rng.normal(0.0, 0.8, 2)
        sim.add_vehicle_pass(feed, start_s=t_s, path_start_xy=gate,
                             path_end_xy=gate,
                             axle_masses=(float(rng.normal(600.0, 15.0)),
                                          float(rng.normal(600.0, 15.0))),
                             label="C")
    elif key == "D":        # an intruder on foot, slow and heavy, never trained
        start = np.array([20.0, 20.0]) + rng.normal(0.0, 1.5, 2)
        end = start + np.array([-6.0, -4.0]) + rng.normal(0.0, 1.0, 2)
        sim.add_footstep_train(feed, start_s=t_s, path_start_xy=start,
                               path_end_xy=end, num_steps=6,
                               cadence_hz=float(rng.normal(1.2, 0.04)),
                               mass_kg=float(rng.normal(130.0, 4.0)),
                               freq_hz=28.0, label="D")
    elif key == "E":        # an intruder vehicle, three heavy axles, never trained
        gate = np.array([15.0, 8.0]) + rng.normal(0.0, 0.8, 2)
        sim.add_vehicle_pass(feed, start_s=t_s, path_start_xy=gate,
                             path_end_xy=gate,
                             axle_masses=(900.0, 900.0, 900.0),
                             axle_spacing_s=0.30, label="E")
    elif key == "noise":    # a one-off oddity (small animal), should stay noise
        start = np.array([14.0, 14.0]) + rng.normal(0.0, 2.0, 2)
        end = start + np.array([4.0, -3.0]) + rng.normal(0.0, 1.5, 2)
        sim.add_footstep_train(feed, start_s=t_s, path_start_xy=start,
                               path_end_xy=end, num_steps=5,
                               cadence_hz=float(rng.normal(3.4, 0.1)),
                               mass_kg=float(rng.normal(40.0, 3.0)),
                               freq_hz=40.0, label="noise")


def _build_stream(keys: Sequence[str], fs: int, seed: int,
                  spacing_s: float = 6.0,
                  lead_s: float = 3.0) -> Tuple[np.ndarray, List[Tuple[float, str]]]:
    """Lay a list of entity appearances onto one feed, evenly spaced in time."""
    rng = np.random.default_rng(seed)
    sim = GeophoneArraySimulator(sampling_rate_hz=fs, seed=seed)
    duration = lead_s + spacing_s * len(keys) + 3.0
    feed = sim.baseline(duration_s=duration)
    truth: List[Tuple[float, str]] = []
    t = lead_s
    for key in keys:
        _emit_entity(sim, feed, key, t, rng)
        truth.append((t, key))
        t += spacing_s
    return feed, truth


def build_learning_stream(fs: int, seed: int) -> Tuple[np.ndarray, List[Tuple[float, str]]]:
    """A month of routine traffic, compressed: three regulars plus one oddity."""
    rng = np.random.default_rng(seed + 100)
    keys = ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["noise"]
    rng.shuffle(keys)
    return _build_stream(keys, fs, seed)


def build_guarding_stream(fs: int, seed: int) -> Tuple[np.ndarray, List[Tuple[float, str]]]:
    """Live traffic: regulars return, plus two never-seen intruders."""
    keys = ["A", "C", "D", "B", "E"]
    return _build_stream(keys, fs, seed + 1)


# ---------------------------------------------------------------------------
# Ground-truth bookkeeping for the post-run check
# ---------------------------------------------------------------------------

def _truth_for_window(window_start_s: float,
                      truth: List[Tuple[float, str]]) -> str:
    """Match a detected window back to the nearest injected event's true key."""
    best_key = "?"
    best_gap = float("inf")
    for event_start, key in truth:
        gap = abs(window_start_s - event_start)
        if gap < best_gap:
            best_gap = gap
            best_key = key
    return best_key


@dataclass
class _Tally:
    by_profile: dict

    def add(self, profile_id: str, truth_key: str) -> None:
        self.by_profile.setdefault(profile_id, {})
        self.by_profile[profile_id][truth_key] = (
            self.by_profile[profile_id].get(truth_key, 0) + 1)


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def run_demo(fs: int = DEFAULT_SAMPLING_RATE_HZ, seed: int = 11,
             match_distance: float = 1.2, enroll_after: int = 3) -> None:
    memory = PatternMemory(FEATURE_SCALES, match_distance=match_distance,
                           enroll_after=enroll_after, feature_names=FEATURE_NAMES)
    # A longer merge gap than the V2 default groups the successive footfalls of
    # one walk-through into a single event, so a slow walker is one signature
    # rather than a string of isolated steps.
    processor = SignalProcessor(fs=fs, merge_gap_s=1.2)

    bar = "=" * 72
    print(bar)
    print("SEISMIC PATTERN MEMORY  |  V1 open core  |  self-learning demonstration")
    print("the system starts knowing nothing; it learns the site from the feed")
    print("match distance {0}  |  enrol after {1} sightings".format(
        match_distance, enroll_after))
    print(bar)

    # -- learning phase ----------------------------------------------------
    raw, truth = build_learning_stream(fs, seed)
    result = processor.process(raw)
    tally = _Tally(by_profile={})

    print("\nLEARNING PHASE  ({0} events arrive, no prior knowledge)\n".format(
        len(result.windows)))
    for window in result.windows:
        vector, _, readable = build_feature_vector(result, window, fs)
        obs = memory.observe(vector, window.start_s)
        tally.add(obs.profile_id, _truth_for_window(window.start_s, truth))

        if obs.is_new:
            print("  t={0:5.1f}s  new signature        -> {1} tentative "
                  "(1 obs)   [cadence {2:.2f} Hz, dom {3:.0f} Hz, "
                  "ratio {4:.2f}]".format(
                      window.start_s, obs.profile_id, readable["cadence_hz"],
                      readable["dominant_freq_hz"], readable["low_freq_ratio"]))
        else:
            tag = "ENROLLED" if obs.newly_enrolled else (
                "enrolled" if obs.enrolled else "tentative")
            note = "  <- confirmed, now enrolled" if obs.newly_enrolled else ""
            print("  t={0:5.1f}s  matched {1} (d={2:.2f})  -> {3} obs, "
                  "{4}{5}".format(
                      window.start_s, obs.profile_id, obs.distance,
                      obs.observations, tag, note))

    enrolled = memory.enrolled_profiles()
    tentative = memory.tentative_profiles()
    print("\n  learning complete: {0} enrolled signature(s), {1} tentative "
          "(treated as noise)".format(len(enrolled), len(tentative)))
    print("\n  enrolled signatures:")
    for profile in enrolled:
        truth_counts = tally.by_profile.get(profile.profile_id, {})
        truth_str = ", ".join("{0}x{1}".format(v, k)
                              for k, v in sorted(truth_counts.items()))
        print("    {0}  {1} obs  [{2}]".format(
            profile.profile_id, profile.count, memory.describe(profile)))
        print("              ground truth in this cluster: {0}".format(truth_str))
    if tentative:
        print("\n  tentative (seen too few times to trust):")
        for profile in tentative:
            truth_counts = tally.by_profile.get(profile.profile_id, {})
            truth_str = ", ".join("{0}x{1}".format(v, k)
                                  for k, v in sorted(truth_counts.items()))
            print("    {0}  {1} obs  [{2}]".format(
                profile.profile_id, profile.count, truth_str))

    # -- guarding phase ----------------------------------------------------
    raw2, truth2 = build_guarding_stream(fs, seed)
    result2 = processor.process(raw2)

    print("\n" + bar)
    print("GUARDING PHASE  (only enrolled signatures are trusted)\n")
    recognised = 0
    flagged = 0
    for window in result2.windows:
        vector, _, readable = build_feature_vector(result2, window, fs)
        match = memory.recognise(vector, window.start_s)
        truth_key = _truth_for_window(window.start_s, truth2)

        if match.recognised:
            recognised += 1
            print("  t={0:5.1f}s  recognised {1} (d={2:.2f})  -> SUPPRESSED   "
                  "(true source {3})".format(
                      window.start_s, match.profile_id, match.distance,
                      truth_key))
        else:
            flagged += 1
            closest = match.closest_id if match.closest_id else "none"
            closest_d = ("{0:.2f}".format(match.distance)
                         if np.isfinite(match.distance) else "inf")
            print("  t={0:5.1f}s  NO MATCH (closest {1} at d={2})  -> "
                  "*** NOVEL SIGNATURE ALERT ***   (true source {3})".format(
                      window.start_s, closest, closest_d, truth_key))

    print("\n" + bar)
    print("summary: {0} recognised and suppressed, {1} novel signature "
          "alert(s)".format(recognised, flagged))
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-learning seismic pattern memory demonstration (V1).")
    parser.add_argument("--fs", type=int, default=DEFAULT_SAMPLING_RATE_HZ,
                        help="sampling rate in Hz (default 100)")
    parser.add_argument("--seed", type=int, default=11,
                        help="random seed for reproducibility (default 11)")
    parser.add_argument("--match-distance", type=float, default=1.2,
                        help="match threshold in feature-scale units (default 1.2)")
    parser.add_argument("--enroll-after", type=int, default=3,
                        help="sightings needed to enrol a signature (default 3)")
    args = parser.parse_args()
    run_demo(fs=args.fs, seed=args.seed, match_distance=args.match_distance,
             enroll_after=args.enroll_after)


if __name__ == "__main__":
    main()

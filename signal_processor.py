"""
signal_processor.py
===================
Real-time-style conditioning for raw geophone voltage feeds.

The processor has two jobs:

1. Band-limit the raw feed to the 10 to 50 Hz seismic band where footfall and
   axle energy lives, rejecting DC offset, slow drift and high-frequency hiss.
   A zero-phase Butterworth filter is used so that event timing is preserved,
   which matters because cadence and axle spacing are timing measurements.

2. Decide, sample by sample, what counts as activity. Instead of a fixed
   trigger level it uses an adaptive threshold built from a rolling median and
   a rolling median absolute deviation of the signal envelope. Because the
   threshold follows the local statistics of the noise floor, slow
   environmental changes such as wind loading, rain or thermal drift raise the
   floor gradually and do not generate false alarms. Short, energetic events
   stand out sharply against that slowly moving baseline.

Depends on numpy and scipy only.

Run directly to process the reference scenario and print the activity windows
that were detected:

    python signal_processor.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, hilbert, sosfiltfilt

from data_simulator import DEFAULT_SAMPLING_RATE_HZ, build_demo_scenario


# A robust constant that scales the median absolute deviation so that, for
# Gaussian data, it estimates the standard deviation.
_MAD_TO_STD = 1.4826


# ---------------------------------------------------------------------------
# Detected activity
# ---------------------------------------------------------------------------

@dataclass
class ActivityWindow:
    """A contiguous stretch of the feed that crossed the adaptive threshold."""

    start_idx: int
    end_idx: int          # exclusive
    start_s: float
    end_s: float
    peak_v: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ProcessResult:
    """Everything the processor produced for one feed."""

    filtered: np.ndarray
    envelope: np.ndarray
    threshold: np.ndarray
    windows: List[ActivityWindow]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def design_bandpass(lowcut: float, highcut: float, fs: int,
                    order: int = 4) -> np.ndarray:
    """Design a Butterworth bandpass as second-order sections.

    Second-order sections are numerically stable at the band edges, which here
    sit close to the Nyquist frequency. The upper edge is clamped just below
    Nyquist (fs / 2) so the design is always valid even when a caller asks for
    the full 50 Hz at a 100 Hz sample rate.
    """
    nyquist = 0.5 * fs
    low = max(lowcut, 0.1) / nyquist
    high = min(highcut, 0.99 * nyquist) / nyquist
    if not 0.0 < low < high < 1.0:
        raise ValueError("invalid band edges for the given sampling rate")
    return butter(order, [low, high], btype="band", output="sos")


def bandpass_filter(signal: np.ndarray, lowcut: float = 10.0,
                    highcut: float = 50.0, fs: int = DEFAULT_SAMPLING_RATE_HZ,
                    order: int = 4) -> np.ndarray:
    """Apply the seismic-band bandpass with zero phase distortion."""
    sos = design_bandpass(lowcut, highcut, fs, order)
    # filtfilt-style forward/backward pass removes group delay so event onsets
    # stay where they actually happened.
    return sosfiltfilt(sos, signal)


# ---------------------------------------------------------------------------
# Envelope and adaptive threshold
# ---------------------------------------------------------------------------

def amplitude_envelope(signal: np.ndarray) -> np.ndarray:
    """Return the analytic-signal amplitude envelope of a band-limited feed."""
    return np.abs(hilbert(signal))


def _odd_window(window_samples: int, n: int) -> int:
    """Coerce a window length to an odd value that fits inside the signal."""
    w = max(3, int(window_samples))
    if w % 2 == 0:
        w += 1
    limit = n if n % 2 == 1 else n - 1
    return max(3, min(w, max(3, limit)))


def adaptive_threshold(envelope: np.ndarray, fs: int,
                       window_s: float = 3.0, k: float = 6.0) -> np.ndarray:
    """Compute a per-sample trigger level that tracks the local noise floor.

    The level is a rolling median of the envelope plus ``k`` robust standard
    deviations, where the spread is estimated from a rolling median absolute
    deviation. Both statistics use a sliding window of ``window_s`` seconds.
    Medians are insensitive to the brief, large excursions that genuine events
    cause, so an event does not inflate its own threshold, while slow drift in
    the floor is tracked faithfully.
    """
    n = envelope.size
    window = _odd_window(int(round(window_s * fs)), n)

    rolling_med = median_filter(envelope, size=window, mode="reflect")
    abs_dev = np.abs(envelope - rolling_med)
    rolling_mad = median_filter(abs_dev, size=window, mode="reflect")

    return rolling_med + k * _MAD_TO_STD * rolling_mad


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_events(envelope: np.ndarray, threshold: np.ndarray, fs: int,
                  min_duration_s: float = 0.1,
                  merge_gap_s: float = 0.5) -> List[Tuple[int, int]]:
    """Find contiguous regions where the envelope exceeds the threshold.

    Regions separated by less than ``merge_gap_s`` are merged so that the
    individual footfalls of a single walker, or the separate axles of one
    vehicle, are reported as a single activity window. Regions shorter than
    ``min_duration_s`` are discarded as transient noise.
    """
    mask = envelope > threshold
    if not mask.any():
        return []

    # Locate rising and falling edges of the boolean mask.
    flags = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], flags, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]  # exclusive

    merge_gap = int(round(merge_gap_s * fs))
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = int(starts[0]), int(ends[0])
    for s, e in zip(starts[1:], ends[1:]):
        if s - cur_end <= merge_gap:
            cur_end = int(e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = int(s), int(e)
    merged.append((cur_start, cur_end))

    min_len = int(round(min_duration_s * fs))
    return [(s, e) for s, e in merged if e - s >= min_len]


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

class SignalProcessor:
    """Convenience wrapper that runs the full conditioning chain on a feed."""

    def __init__(self, fs: int = DEFAULT_SAMPLING_RATE_HZ,
                 lowcut: float = 10.0, highcut: float = 50.0,
                 filter_order: int = 4, threshold_window_s: float = 3.0,
                 threshold_k: float = 6.0, min_duration_s: float = 0.1,
                 merge_gap_s: float = 0.5) -> None:
        self.fs = int(fs)
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.threshold_window_s = threshold_window_s
        self.threshold_k = threshold_k
        self.min_duration_s = min_duration_s
        self.merge_gap_s = merge_gap_s

    def process(self, raw: np.ndarray) -> ProcessResult:
        """Filter, build the envelope and adaptive threshold, detect windows."""
        filtered = bandpass_filter(raw, self.lowcut, self.highcut, self.fs,
                                   self.filter_order)
        envelope = amplitude_envelope(filtered)
        threshold = adaptive_threshold(envelope, self.fs,
                                       self.threshold_window_s, self.threshold_k)
        spans = detect_events(envelope, threshold, self.fs,
                              self.min_duration_s, self.merge_gap_s)

        windows: List[ActivityWindow] = []
        for start, end in spans:
            windows.append(ActivityWindow(
                start_idx=start, end_idx=end,
                start_s=start / self.fs, end_s=end / self.fs,
                peak_v=float(envelope[start:end].max())))

        return ProcessResult(filtered=filtered, envelope=envelope,
                             threshold=threshold, windows=windows)


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fs = DEFAULT_SAMPLING_RATE_HZ
    raw = build_demo_scenario(fs=fs, seed=7)
    result = SignalProcessor(fs=fs).process(raw)

    print("Detected {0} activity window(s):".format(len(result.windows)))
    print("  {0:>8}  {1:>8}  {2:>10}".format("start_s", "end_s", "peak_v"))
    for w in result.windows:
        print("  {0:8.2f}  {1:8.2f}  {2:10.4f}".format(
            w.start_s, w.end_s, w.peak_v))


if __name__ == "__main__":
    main()

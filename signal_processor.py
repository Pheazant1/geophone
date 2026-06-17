"""
signal_processor.py
===================
Real-time-style conditioning for the raw geophone array feed.

The processor has three jobs:

1. Band-limit every channel to the 10 to 50 Hz seismic band where footfall and
   axle energy lives, rejecting DC offset, slow drift and high-frequency hiss.
   A zero-phase Butterworth filter is used so event timing is preserved across
   channels, which matters because the localiser keys on inter-channel timing.

2. Decide, sample by sample, what counts as activity. Instead of a fixed
   trigger it uses an adaptive threshold built from a rolling median and a
   rolling median absolute deviation of the array-averaged envelope. Because
   the threshold follows the local statistics of the noise floor, slow
   environmental changes such as wind, rain or thermal drift raise the floor
   gradually and do not generate false alarms.

3. Group threshold crossings into activity windows shared across the array, and
   tag each window with the channel that saw it most strongly (the closest
   sensor, which has the best signal to noise for feature extraction).

Depends on numpy and scipy only.

Run directly to process the reference scenario and print the activity windows:

    python signal_processor.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, hilbert, sosfiltfilt

from data_simulator import DEFAULT_SAMPLING_RATE_HZ, build_demo_scenario


# Scales the median absolute deviation to estimate a standard deviation for
# Gaussian data.
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
    peak_v: float         # peak envelope on the strongest channel
    best_channel: int     # channel with the strongest response (closest sensor)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ProcessResult:
    """Everything the processor produced for one array feed."""

    filtered: np.ndarray            # (n_channels, n_samples)
    envelope: np.ndarray            # (n_channels, n_samples)
    detection_envelope: np.ndarray  # (n_samples,) array-averaged envelope
    threshold: np.ndarray           # (n_samples,)
    windows: List[ActivityWindow]


# ---------------------------------------------------------------------------
# Filtering (operates on a single channel)
# ---------------------------------------------------------------------------

def design_bandpass(lowcut: float, highcut: float, fs: int,
                    order: int = 4) -> np.ndarray:
    """Design a Butterworth bandpass as numerically stable second-order sections.

    The upper edge is clamped just below Nyquist (fs / 2) so the design is valid
    even when a caller asks for the full 50 Hz at a 100 Hz sample rate.
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
    """Apply the seismic-band bandpass to one channel with zero phase shift."""
    sos = design_bandpass(lowcut, highcut, fs, order)
    return sosfiltfilt(sos, signal)


def amplitude_envelope(signal: np.ndarray) -> np.ndarray:
    """Return the analytic-signal amplitude envelope of one band-limited channel."""
    return np.abs(hilbert(signal))


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

def _odd_window(window_samples: int, n: int) -> int:
    w = max(3, int(window_samples))
    if w % 2 == 0:
        w += 1
    limit = n if n % 2 == 1 else n - 1
    return max(3, min(w, max(3, limit)))


def adaptive_threshold(envelope: np.ndarray, fs: int,
                       window_s: float = 3.0, k: float = 6.0) -> np.ndarray:
    """Per-sample trigger level that tracks the local noise floor.

    The level is a rolling median of the envelope plus ``k`` robust standard
    deviations estimated from a rolling median absolute deviation. Medians
    ignore the brief, large excursions of genuine events, so an event never
    inflates its own threshold, while slow drift in the floor is tracked.
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

    Regions separated by less than ``merge_gap_s`` are merged, so the footfalls
    of one walker or the axles of one vehicle form a single window. Regions
    shorter than ``min_duration_s`` are discarded as transient noise.
    """
    mask = envelope > threshold
    if not mask.any():
        return []

    flags = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], flags, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

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
    """Runs the full conditioning chain on a multi-channel array feed."""

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
        """Filter every channel, then detect activity on the array average.

        ``raw`` is shaped (n_channels, n_samples). A single-channel 1-D input is
        accepted and treated as a one-element array.
        """
        if raw.ndim == 1:
            raw = raw[None, :]
        n_channels = raw.shape[0]

        filtered = np.empty_like(raw, dtype=float)
        envelope = np.empty_like(raw, dtype=float)
        for i in range(n_channels):
            filtered[i] = bandpass_filter(raw[i], self.lowcut, self.highcut,
                                          self.fs, self.filter_order)
            envelope[i] = amplitude_envelope(filtered[i])

        # Detect on the array-averaged envelope: an event anywhere inside the
        # footprint raises all channels together, so the mean is a stable,
        # high signal-to-noise trigger.
        detection_envelope = envelope.mean(axis=0)
        threshold = adaptive_threshold(detection_envelope, self.fs,
                                       self.threshold_window_s, self.threshold_k)
        spans = detect_events(detection_envelope, threshold, self.fs,
                              self.min_duration_s, self.merge_gap_s)

        windows: List[ActivityWindow] = []
        for start, end in spans:
            channel_peaks = envelope[:, start:end].max(axis=1)
            best_channel = int(np.argmax(channel_peaks))
            windows.append(ActivityWindow(
                start_idx=start, end_idx=end,
                start_s=start / self.fs, end_s=end / self.fs,
                peak_v=float(channel_peaks[best_channel]),
                best_channel=best_channel))

        return ProcessResult(filtered=filtered, envelope=envelope,
                             detection_envelope=detection_envelope,
                             threshold=threshold, windows=windows)


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fs = DEFAULT_SAMPLING_RATE_HZ
    raw = build_demo_scenario(fs=fs, seed=7)
    result = SignalProcessor(fs=fs).process(raw)

    print("Detected {0} activity window(s) across {1} channels:".format(
        len(result.windows), result.filtered.shape[0]))
    print("  {0:>8}  {1:>8}  {2:>10}  {3:>8}".format(
        "start_s", "end_s", "peak_v", "channel"))
    for w in result.windows:
        print("  {0:8.2f}  {1:8.2f}  {2:10.4f}  {3:8d}".format(
            w.start_s, w.end_s, w.peak_v, w.best_channel))


if __name__ == "__main__":
    main()

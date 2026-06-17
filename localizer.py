"""
localizer.py
============
Source localisation and heading estimation for the geophone array.

When the same ground disturbance reaches four sensors at different times, those
time differences pin down where the disturbance happened. This module measures
the time differences of arrival (TDOA) between channels by cross-correlation,
then solves for the source position by searching the protected area for the
point whose predicted delays best match what was observed. For an event that
lasts long enough, it localises the early and late parts separately and reports
a heading and a speed, turning a static detection into a track.

The physics is the standard surface-wave relation: travel time equals distance
over wave speed. With four sensors and three independent time differences the
position is well determined inside the array footprint.

Depends on numpy and scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import correlate

from data_simulator import SENSOR_POSITIONS_M, WAVE_SPEED_M_S


@dataclass
class Localization:
    """An estimated source position with a fit-quality residual."""

    x_m: float
    y_m: float
    residual_s: float            # root-mean-square TDOA misfit, seconds


@dataclass
class Track:
    """A localised event, with heading and speed when they can be resolved."""

    position: Localization
    heading_deg: Optional[float] = None        # compass bearing, 0 = +y, clockwise
    speed_m_s: Optional[float] = None
    start_xy: Optional[Tuple[float, float]] = None
    end_xy: Optional[Tuple[float, float]] = None


# ---------------------------------------------------------------------------
# Time difference of arrival
# ---------------------------------------------------------------------------

def estimate_tdoa(filtered: np.ndarray, start_idx: int, end_idx: int, fs: int,
                  ref_channel: int = 0, max_lag_s: float = 0.2) -> np.ndarray:
    """Return the per-channel arrival delay relative to ``ref_channel``.

    Each channel is cross-correlated against the reference channel over the
    event window. The lag of the correlation peak is the time difference of
    arrival, refined to sub-sample resolution by fitting a parabola to the
    three samples around the peak. The lag search is restricted to physically
    possible delays, which also stops the periodic footfalls of a walker from
    locking onto the wrong cycle.
    """
    seg = filtered[:, start_idx:end_idx]
    n_channels, seg_len = seg.shape
    max_lag = int(round(max_lag_s * fs))
    ref = seg[ref_channel] - seg[ref_channel].mean()

    tdoa = np.zeros(n_channels)
    for i in range(n_channels):
        sig = seg[i] - seg[i].mean()
        corr = correlate(sig, ref, mode="full")
        lags = np.arange(-seg_len + 1, seg_len)
        keep = np.abs(lags) <= max_lag
        corr_w = corr[keep]
        lags_w = lags[keep]
        k = int(np.argmax(corr_w))

        # Parabolic interpolation for a fractional-sample peak position.
        if 0 < k < corr_w.size - 1:
            y0, y1, y2 = corr_w[k - 1], corr_w[k], corr_w[k + 1]
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        else:
            delta = 0.0
        tdoa[i] = (lags_w[k] + delta) / fs

    return tdoa - tdoa[ref_channel]


# ---------------------------------------------------------------------------
# Position from TDOA
# ---------------------------------------------------------------------------

def localize(tdoa: np.ndarray,
             sensor_positions: np.ndarray = SENSOR_POSITIONS_M,
             wave_speed_m_s: float = WAVE_SPEED_M_S,
             ref_channel: int = 0, grid_step_m: float = 0.5,
             margin_m: float = 5.0) -> Localization:
    """Solve for the source position whose predicted TDOA best fits ``tdoa``.

    A regular grid is laid over the array footprint (expanded by ``margin_m``)
    and, for every grid point, the predicted TDOA relative to the reference
    sensor is compared with the measurement. The grid point with the smallest
    misfit is returned, along with the root-mean-square residual as a
    confidence indicator.
    """
    positions = np.asarray(sensor_positions, dtype=float)
    x_min, y_min = positions.min(axis=0) - margin_m
    x_max, y_max = positions.max(axis=0) + margin_m
    xs = np.arange(x_min, x_max + grid_step_m, grid_step_m)
    ys = np.arange(y_min, y_max + grid_step_m, grid_step_m)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)   # (G, 2)

    # Predicted delay from every grid point to every sensor, made relative to
    # the reference sensor so the unknown emission time cancels out.
    dist = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
    pred = dist / wave_speed_m_s
    pred = pred - pred[:, ref_channel:ref_channel + 1]

    misfit = np.sum((pred - tdoa[None, :]) ** 2, axis=1)
    best = int(np.argmin(misfit))
    residual = float(np.sqrt(misfit[best] / positions.shape[0]))
    return Localization(x_m=round(float(points[best, 0]), 2),
                        y_m=round(float(points[best, 1]), 2),
                        residual_s=residual)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def _bearing_and_speed(start: Localization, end: Localization,
                       dt_s: float) -> Tuple[Optional[float], Optional[float]]:
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    distance = float(np.hypot(dx, dy))
    if distance < 1.0 or dt_s <= 0:
        # Too little displacement to call a direction reliably.
        return None, None
    bearing = float(np.degrees(np.arctan2(dx, dy))) % 360.0
    return round(bearing, 1), round(distance / dt_s, 2)


def track_window(filtered: np.ndarray, start_idx: int, end_idx: int, fs: int,
                 sensor_positions: np.ndarray = SENSOR_POSITIONS_M,
                 wave_speed_m_s: float = WAVE_SPEED_M_S,
                 min_track_duration_s: float = 1.5) -> Track:
    """Localise an event window and, if it is long enough, add a heading.

    The whole-window solution gives the representative position. When the event
    spans at least ``min_track_duration_s`` the leading and trailing thirds are
    localised separately; the vector between them yields the direction of travel
    and an average speed.
    """
    tdoa = estimate_tdoa(filtered, start_idx, end_idx, fs)
    position = localize(tdoa, sensor_positions, wave_speed_m_s)

    duration_s = (end_idx - start_idx) / fs
    if duration_s < min_track_duration_s:
        return Track(position=position)

    third = (end_idx - start_idx) // 3
    early = localize(estimate_tdoa(filtered, start_idx, start_idx + third, fs),
                     sensor_positions, wave_speed_m_s)
    late = localize(estimate_tdoa(filtered, end_idx - third, end_idx, fs),
                    sensor_positions, wave_speed_m_s)
    dt = (2.0 / 3.0) * duration_s   # centre-to-centre spacing of the two thirds
    bearing, speed = _bearing_and_speed(early, late, dt)

    return Track(position=position, heading_deg=bearing, speed_m_s=speed,
                 start_xy=(early.x_m, early.y_m), end_xy=(late.x_m, late.y_m))


def heading_to_compass(bearing_deg: Optional[float]) -> Optional[str]:
    """Convert a bearing in degrees to an eight-point compass label."""
    if bearing_deg is None:
        return None
    points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return points[int((bearing_deg + 22.5) % 360 // 45)]

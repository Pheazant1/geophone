"""
environment.py
==============
Ground and weather presets for the simulation game.

Seismic ground response is extremely site-specific, so none of these numbers is
a universal truth; they are physically reasonable settings with the right
direction of effect. Every field is tagged as either "grounded" (the direction
and rough range are supported by published soil and gait physics) or "estimate"
(a plausible value that a real deployment would replace with a calibration
measurement). The game shows these tags so it is always clear which knobs are
real and which are educated guesses.

Each preset feeds the existing physics: wave speed and absorption go straight
into the array simulator and the localiser, the noise floor sets the background,
and the coupling gain scales how strongly a given dynamic load couples into the
ground (the part a real site would have to calibrate and the reason the same
person can look different in different weather).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Environment:
    """A ground-and-weather condition and how it changes the seismic physics."""

    key: str
    name: str
    summary: str
    wave_speed_m_s: float       # surface-wave speed in the soil
    absorption_per_m: float     # anelastic absorption, higher means faster decay
    noise_floor_v: float        # background noise standard deviation per channel
    coupling_gain: float        # how strongly a given load couples in vs baseline
    # confidence per field: "grounded" or "estimate", with a short reason
    confidence: Dict[str, Tuple[str, str]]


PRESETS: Dict[str, Environment] = {
    "dry_firm": Environment(
        key="dry_firm", name="Dry firm soil", summary="calm baseline conditions",
        wave_speed_m_s=250.0, absorption_per_m=0.010, noise_floor_v=0.0015,
        coupling_gain=1.00,
        confidence={
            "wave_speed_m_s": ("grounded", "firm soils sit in the low hundreds m/s"),
            "absorption_per_m": ("estimate", "soil Q is poorly constrained"),
            "noise_floor_v": ("estimate", "site-specific background"),
            "coupling_gain": ("grounded", "reference case, gain defined as 1.0"),
        }),
    "wet": Environment(
        key="wet", name="Wet soil", summary="recent rain, soil damp",
        wave_speed_m_s=270.0, absorption_per_m=0.012, noise_floor_v=0.0018,
        coupling_gain=1.10,
        confidence={
            "wave_speed_m_s": ("grounded", "saturation tends to raise wave speed"),
            "absorption_per_m": ("estimate", "moderate guess"),
            "noise_floor_v": ("estimate", "slightly higher than dry"),
            "coupling_gain": ("estimate", "wet ground often transmits low band better"),
        }),
    "mud": Environment(
        key="mud", name="Mud / soft saturated", summary="soft surface, heavy damping",
        wave_speed_m_s=150.0, absorption_per_m=0.030, noise_floor_v=0.0020,
        coupling_gain=0.55,
        confidence={
            "wave_speed_m_s": ("grounded", "soft saturated ground is slow"),
            "absorption_per_m": ("estimate", "high damping, magnitude is a guess"),
            "noise_floor_v": ("estimate", "site-specific"),
            "coupling_gain": ("estimate", "mud muffles the signal, amount uncertain"),
        }),
    "snow": Environment(
        key="snow", name="Snow cover", summary="snow absorbs high frequencies",
        wave_speed_m_s=220.0, absorption_per_m=0.040, noise_floor_v=0.0012,
        coupling_gain=0.50,
        confidence={
            "wave_speed_m_s": ("estimate", "depends on snow depth and pack"),
            "absorption_per_m": ("estimate", "snow is a strong absorber, magnitude guessed"),
            "noise_floor_v": ("estimate", "snow tends to quieten the surface"),
            "coupling_gain": ("estimate", "surface coupling reduced, amount uncertain"),
        }),
    "frozen": Environment(
        key="frozen", name="Frozen ground", summary="stiff, fast, signals travel far",
        wave_speed_m_s=600.0, absorption_per_m=0.004, noise_floor_v=0.0013,
        coupling_gain=1.30,
        confidence={
            "wave_speed_m_s": ("grounded", "frozen ground is much stiffer and faster"),
            "absorption_per_m": ("grounded", "low loss, signals carry farther"),
            "noise_floor_v": ("estimate", "site-specific"),
            "coupling_gain": ("estimate", "stiff coupling preserves more signal"),
        }),
    "rain": Environment(
        key="rain", name="Active rainfall", summary="raindrops raise the noise floor",
        wave_speed_m_s=270.0, absorption_per_m=0.012, noise_floor_v=0.0040,
        coupling_gain=1.05,
        confidence={
            "wave_speed_m_s": ("grounded", "wet, like the wet-soil case"),
            "absorption_per_m": ("estimate", "moderate guess"),
            "noise_floor_v": ("grounded", "raindrop impacts clearly raise broadband noise"),
            "coupling_gain": ("estimate", "small change over wet soil"),
        }),
}


PRESET_ORDER = ["dry_firm", "wet", "mud", "snow", "frozen", "rain"]


def default_environment() -> Environment:
    return PRESETS["dry_firm"]


def next_preset(current_key: str) -> Environment:
    """Cycle to the next preset, for a single click to step through conditions."""
    idx = PRESET_ORDER.index(current_key) if current_key in PRESET_ORDER else 0
    return PRESETS[PRESET_ORDER[(idx + 1) % len(PRESET_ORDER)]]

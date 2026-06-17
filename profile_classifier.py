"""
profile_classifier.py
======================
Decision engine for the Seismic Perimeter Intelligence core.

This stage turns a conditioned, localised activity window into an operational
decision. It

  1. extracts physical features from the strongest channel (impact cadence or
     axle spacing, spectral character, per-impact amplitudes),
  2. classifies the source as human, vehicle or wildlife,
  3. localises the source and, for moving targets, recovers a heading and speed,
  4. estimates an effective mass from the peak amplitude, range-corrected using
     the localised distance to the sensor, via the linear seismic coupling of
     the site,
  5. compares the result against saved profiles for known people and vehicles,
     and
  6. emits a structured JSON alert webhook for anything unknown, or for a known
     target whose physical profile has shifted in a way that matters (for
     example a registered vehicle that returns with significantly more load on
     one axle).

Recognised, unchanged traffic is suppressed. Because every alert carries a
position, the recommended camera response can be aimed at the exact spot before
the target is in frame.

Depends on numpy and scipy for analysis and the Python standard library for the
alert payload. Run directly for an end-to-end demonstration:

    python profile_classifier.py
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from data_simulator import (ABSORPTION_PER_M, DEFAULT_SAMPLING_RATE_HZ,
                            REF_DISTANCE_M, SENSOR_POSITIONS_M,
                            build_demo_scenario)
from localizer import Localization, Track, heading_to_compass, track_window
from signal_processor import ActivityWindow, ProcessResult, SignalProcessor


# ---------------------------------------------------------------------------
# Site calibration
# ---------------------------------------------------------------------------
#
# End-to-end calibration constants: processed volts of envelope peak per
# kilogram of effective load, referenced to the calibration distance. They fold
# together the physical ground coupling and the passband gain of the
# conditioning chain, and are measured during commissioning by walking a known
# mass and driving a known axle load across the array at a known location. A
# live amplitude is range-corrected back to this reference distance before the
# mass is read off.

PEDESTRIAN_V_PER_KG_AT_REF: float = 7.8e-4
AXLE_V_PER_KG_AT_REF: float = 8.5e-5

# Padding applied around a detection window before counting impacts, so a
# leading footfall or axle that sits right on the window edge is not clipped.
FEATURE_PAD_S: float = 0.3


# ---------------------------------------------------------------------------
# Sensor array identity (would come from device configuration in the field)
# ---------------------------------------------------------------------------

SENSOR_ARRAY: Dict[str, object] = {
    "node_id": "GP-NODE-01",
    "array": "PERIMETER-WEST",
    "channels": int(SENSOR_POSITIONS_M.shape[0]),
    "sensor_positions_m": SENSOR_POSITIONS_M.tolist(),
}


# ---------------------------------------------------------------------------
# Saved profiles of recognised people and assets
# ---------------------------------------------------------------------------

SAVED_PROFILES: Dict[str, Dict[str, object]] = {
    "OWN-PED-01": {
        "class": "human",
        "label": "Property owner (enrolled, on foot)",
        "cadence_hz": 1.8,
        "body_mass_kg": 75.0,
        "tolerance": {"cadence_pct": 0.20, "mass_pct": 0.15},
    },
    "OWN-VEH-01": {
        "class": "vehicle",
        "label": "Owner pickup (registered, baseline unladen)",
        "num_axles": 2,
        "axle_spacing_s": 0.45,
        "wheelbase_m": 3.2,
        "heaviest_axle_kg": 650.0,
        "tolerance": {"spacing_pct": 0.30, "axle_mass_pct": 0.10},
    },
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class Features:
    """Physical descriptors recovered from one activity window."""

    peak_v: float
    heaviest_impact_v: float
    impact_heights_v: List[float]
    num_impacts: int
    median_interval_s: float
    interval_cv: float
    cadence_hz: float
    dominant_freq_hz: float
    low_freq_ratio: float
    heaviest_impact_index: int


def _spectral_features(segment: np.ndarray, fs: int,
                       band: Tuple[float, float] = (10.0, 45.0),
                       low_split_hz: float = 18.0) -> Tuple[float, float]:
    """Return (dominant frequency, low-frequency energy ratio) for a segment."""
    n = segment.size
    if n < 8:
        return 0.0, 0.0
    window = np.hanning(n)
    power = np.abs(np.fft.rfft(segment * window)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any() or power[in_band].sum() == 0.0:
        return 0.0, 0.0
    band_freqs = freqs[in_band]
    band_power = power[in_band]
    dominant = float(band_freqs[int(np.argmax(band_power))])
    low_ratio = float(band_power[band_freqs <= low_split_hz].sum() / band_power.sum())
    return dominant, low_ratio


def extract_features(result: ProcessResult, window: ActivityWindow,
                     fs: int) -> Features:
    """Recover footfall or axle structure and spectral character.

    Features are taken from the strongest channel, which is the sensor closest
    to the source and therefore has the best signal to noise. The analysis
    window is padded slightly so an impact on the boundary is not missed.
    """
    ch = window.best_channel
    pad = int(round(FEATURE_PAD_S * fs))
    s = max(0, window.start_idx - pad)
    e = min(result.envelope.shape[1], window.end_idx + pad)
    env = result.envelope[ch][s:e]
    seg = result.filtered[ch][s:e]
    peak_v = float(env.max()) if env.size else 0.0

    peaks, props = find_peaks(
        env, height=0.35 * peak_v,
        distance=max(1, int(round(0.18 * fs))),
        prominence=0.2 * peak_v)
    heights = props["peak_heights"] if peaks.size else np.array([peak_v])

    if peaks.size >= 2:
        intervals = np.diff(peaks) / fs
        median_interval = float(np.median(intervals))
        interval_cv = float(np.std(intervals) / median_interval) if median_interval > 0 else 0.0
    else:
        median_interval = 0.0
        interval_cv = 0.0

    cadence_hz = (1.0 / median_interval) if median_interval > 0 else 0.0
    dominant_freq, low_ratio = _spectral_features(seg, fs)

    return Features(
        peak_v=peak_v,
        heaviest_impact_v=float(heights.max()),
        impact_heights_v=[round(float(h), 4) for h in heights],
        num_impacts=int(peaks.size) if peaks.size else 1,
        median_interval_s=median_interval,
        interval_cv=interval_cv,
        cadence_hz=cadence_hz,
        dominant_freq_hz=dominant_freq,
        low_freq_ratio=low_ratio,
        heaviest_impact_index=int(np.argmax(heights)))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(features: Features) -> Tuple[str, float]:
    """Return (class, confidence) from distance-independent features.

    A vehicle couples most of its energy into the low part of the band through
    heavy, slow axle deflections and engine rumble, so it is low-frequency
    dominated. A walker produces sharper, higher-frequency footfalls at a steady
    cadence. Amplitude is deliberately not used here because it depends on range
    to the sensor; the detector has already rejected anything at the noise
    floor. Anything that fits neither pattern is treated as wildlife or
    environmental and is not actionable.
    """
    dom = features.dominant_freq_hz
    low_ratio = features.low_freq_ratio
    cadence = features.cadence_hz

    if dom < 20.0 and low_ratio > 0.45:
        return "vehicle", round(min(0.99, 0.65 + 0.34 * low_ratio), 3)

    if 20.0 <= dom <= 42.0 and 1.2 <= cadence <= 3.0:
        regularity = max(0.0, 1.0 - min(features.interval_cv, 1.0))
        return "human", round(min(0.99, 0.60 + 0.38 * regularity), 3)

    return "wildlife", 0.5


# ---------------------------------------------------------------------------
# Range-corrected mass estimation
# ---------------------------------------------------------------------------

def _amplitude_factor(distance_m: float) -> float:
    """Surface-wave amplitude factor at a range, normalised to the reference."""
    d = max(distance_m, REF_DISTANCE_M)
    return float(np.sqrt(REF_DISTANCE_M / d) * np.exp(-ABSORPTION_PER_M * (d - REF_DISTANCE_M)))


def estimate_mass(classification: str, features: Features,
                  best_channel: int, position: Localization) -> Dict[str, float]:
    """Convert peak amplitude into effective mass, corrected for range.

    The measured amplitude is first divided by the distance attenuation between
    the source and the sensor that recorded it, recovering the amplitude the
    impact would have produced at the calibration distance. The linear coupling
    then maps that to a mass.
    """
    sensor = SENSOR_POSITIONS_M[best_channel]
    distance = float(np.hypot(sensor[0] - position.x_m, sensor[1] - position.y_m))
    factor = _amplitude_factor(distance)

    if classification == "vehicle":
        per_axle = [(h / factor) / AXLE_V_PER_KG_AT_REF for h in features.impact_heights_v]
        heaviest = (features.heaviest_impact_v / factor) / AXLE_V_PER_KG_AT_REF
        return {"heaviest_axle_kg": round(heaviest, 1),
                "gross_estimate_kg": round(float(sum(per_axle)), 1),
                "range_m": round(distance, 1)}

    reference_amp = features.heaviest_impact_v / factor
    if classification == "human":
        return {"body_mass_kg": round(reference_amp / PEDESTRIAN_V_PER_KG_AT_REF, 1),
                "range_m": round(distance, 1)}
    return {"effective_kg": round(reference_amp / PEDESTRIAN_V_PER_KG_AT_REF, 1),
            "range_m": round(distance, 1)}


# ---------------------------------------------------------------------------
# Profile matching and decision
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    status: str                                  # authorized | anomalous | unknown | ignored
    threat_level: str                            # INFO | LOW | ELEVATED | HIGH
    emit_alert: bool
    matched_profile_id: Optional[str] = None
    matched_label: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    deviations: Dict[str, object] = field(default_factory=dict)


def _within(value: float, reference: float, tolerance_pct: float) -> bool:
    if reference == 0:
        return False
    return abs(value - reference) / reference <= tolerance_pct


def _evaluate_human(features: Features, mass: Dict[str, float]) -> Decision:
    body_mass = mass["body_mass_kg"]
    for pid, profile in SAVED_PROFILES.items():
        if profile["class"] != "human":
            continue
        tol = profile["tolerance"]
        if not _within(features.cadence_hz, profile["cadence_hz"], tol["cadence_pct"]):
            continue
        if _within(body_mass, profile["body_mass_kg"], tol["mass_pct"]):
            return Decision(
                status="authorized", threat_level="INFO", emit_alert=False,
                matched_profile_id=pid, matched_label=profile["label"],
                reasons=["gait and mass match enrolled profile"])
        delta = (body_mass - profile["body_mass_kg"]) / profile["body_mass_kg"]
        return Decision(
            status="anomalous", threat_level="ELEVATED", emit_alert=True,
            matched_profile_id=pid, matched_label=profile["label"],
            reasons=["gait matches enrolled profile but body mass deviates by "
                     "{0:+.0%}, consistent with a carried load".format(delta)],
            deviations={"body_mass_pct": round(delta, 3),
                        "expected_kg": profile["body_mass_kg"],
                        "measured_kg": body_mass})
    return Decision(
        status="unknown", threat_level="HIGH", emit_alert=True,
        reasons=["pedestrian gait does not match any enrolled profile"])


def _evaluate_vehicle(features: Features, mass: Dict[str, float]) -> Decision:
    measured_heaviest = mass["heaviest_axle_kg"]
    for pid, profile in SAVED_PROFILES.items():
        if profile["class"] != "vehicle":
            continue
        tol = profile["tolerance"]
        axles_ok = features.num_impacts == profile["num_axles"]
        spacing_ok = _within(features.median_interval_s,
                             profile["axle_spacing_s"], tol["spacing_pct"])
        if not (axles_ok and spacing_ok):
            continue
        if _within(measured_heaviest, profile["heaviest_axle_kg"], tol["axle_mass_pct"]):
            return Decision(
                status="authorized", threat_level="INFO", emit_alert=False,
                matched_profile_id=pid, matched_label=profile["label"],
                reasons=["axle geometry and load match registered profile"])
        delta = (measured_heaviest - profile["heaviest_axle_kg"]) / profile["heaviest_axle_kg"]
        axle_pos = "trailing" if features.heaviest_impact_index >= features.num_impacts - 1 else "leading"
        return Decision(
            status="anomalous", threat_level="ELEVATED", emit_alert=True,
            matched_profile_id=pid, matched_label=profile["label"],
            reasons=["registered vehicle returned with {0:+.0%} load change on "
                     "the {1} axle, consistent with added payload".format(delta, axle_pos)],
            deviations={"axle_load_pct": round(delta, 3),
                        "axle_position": axle_pos,
                        "expected_kg": profile["heaviest_axle_kg"],
                        "measured_kg": measured_heaviest})
    return Decision(
        status="unknown", threat_level="HIGH", emit_alert=True,
        reasons=["axle count or spacing does not match any registered vehicle"])


def evaluate(classification: str, features: Features,
             mass: Dict[str, float]) -> Decision:
    if classification == "human":
        return _evaluate_human(features, mass)
    if classification == "vehicle":
        return _evaluate_vehicle(features, mass)
    return Decision(status="ignored", threat_level="LOW", emit_alert=False,
                    reasons=["sub-threshold or irregular, classified as wildlife"])


# ---------------------------------------------------------------------------
# Camera tasking from a localised position
# ---------------------------------------------------------------------------

def select_camera(x_m: float, y_m: float) -> Tuple[str, int]:
    """Pick the camera and PTZ preset that covers a localised position.

    The footprint is split into four quadrants, each served by one camera, so a
    seismic detection can cue the right camera to the right corner of the site.
    """
    centre = SENSOR_POSITIONS_M.mean(axis=0)
    east = x_m >= centre[0]
    north = y_m >= centre[1]
    if north and east:
        return "CAM-NE-01", 1
    if north and not east:
        return "CAM-NW-02", 2
    if not north and not east:
        return "CAM-SW-03", 3
    return "CAM-SE-04", 4


# ---------------------------------------------------------------------------
# Alert payload
# ---------------------------------------------------------------------------

def build_alert_payload(window: ActivityWindow, classification: str,
                        confidence: float, features: Features,
                        mass: Dict[str, float], track: Track,
                        decision: Decision) -> Dict:
    """Assemble the JSON webhook payload a VMS or camera platform consumes."""
    kinematics: Dict[str, object] = {
        "impacts": features.num_impacts,
        "dominant_frequency_hz": round(features.dominant_freq_hz, 2),
        "low_frequency_ratio": round(features.low_freq_ratio, 3),
    }
    if classification == "human":
        kinematics["cadence_hz"] = round(features.cadence_hz, 2)
        kinematics["gait_regularity"] = round(max(0.0, 1.0 - features.interval_cv), 3)
    elif classification == "vehicle":
        kinematics["axle_spacing_s"] = round(features.median_interval_s, 3)

    localization = {
        "x_m": track.position.x_m,
        "y_m": track.position.y_m,
        "residual_s": round(track.position.residual_s, 4),
        "range_m": mass.get("range_m"),
        "heading_deg": track.heading_deg,
        "heading_compass": heading_to_compass(track.heading_deg),
        "speed_m_s": track.speed_m_s,
    }

    dispatch = decision.threat_level in ("ELEVATED", "HIGH")
    camera_id, preset = select_camera(track.position.x_m, track.position.y_m)
    response = {
        "dispatch": dispatch,
        "vms_action": "GOTO_PRESET" if dispatch else "LOG_ONLY",
        "camera_id": camera_id if dispatch else None,
        "ptz_preset": preset if dispatch else None,
        "notify_channels": ["soc-webhook", "guard-mobile"] if dispatch else [],
    }

    mass_payload = {"method": "range_corrected_linear_coupling"}
    mass_payload.update(mass)

    return {
        "schema": "seismic.alert.v1",
        "event_id": str(uuid.uuid4()),
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "sensor": dict(SENSOR_ARRAY),
        "detection": {
            "window": {
                "start_s": round(window.start_s, 2),
                "end_s": round(window.end_s, 2),
                "duration_s": round(window.duration_s, 2),
            },
            "classification": classification,
            "confidence": confidence,
            "strongest_channel": window.best_channel,
            "peak_amplitude_v": round(features.peak_v, 4),
        },
        "localization": localization,
        "kinematics": kinematics,
        "mass_estimate": mass_payload,
        "identity": {
            "status": decision.status,
            "matched_profile": decision.matched_profile_id,
            "label": decision.matched_label,
            "deviations": decision.deviations,
        },
        "threat": {"level": decision.threat_level, "reasons": decision.reasons},
        "response": response,
    }


# ---------------------------------------------------------------------------
# End-to-end demonstration
# ---------------------------------------------------------------------------

def run_pipeline(fs: int = DEFAULT_SAMPLING_RATE_HZ, seed: int = 7) -> None:
    """Simulate, condition, localise, classify and decide over the scenario."""
    raw = build_demo_scenario(fs=fs, seed=seed)
    result = SignalProcessor(fs=fs).process(raw)

    print("=" * 70)
    print("SEISMIC PERIMETER INTELLIGENCE  |  live decision stream")
    print("node {0} / array {1} / {2} channels / {3} activity windows".format(
        SENSOR_ARRAY["node_id"], SENSOR_ARRAY["array"],
        SENSOR_ARRAY["channels"], len(result.windows)))
    print("=" * 70)

    alerts = 0
    suppressed = 0
    for window in result.windows:
        features = extract_features(result, window, fs)
        track = track_window(result.filtered, window.start_idx, window.end_idx, fs)
        classification, confidence = classify(features)
        mass = estimate_mass(classification, features, window.best_channel,
                             track.position)
        decision = evaluate(classification, features, mass)

        where = "({0:.1f}, {1:.1f}) m".format(track.position.x_m, track.position.y_m)
        heading = heading_to_compass(track.heading_deg)
        heading_text = " heading {0}".format(heading) if heading else ""

        if decision.emit_alert:
            alerts += 1
            payload = build_alert_payload(window, classification, confidence,
                                          features, mass, track, decision)
            print("\nALERT  t={0:.2f}s  {1}  at {2}{3}  [{4}]".format(
                window.start_s, classification.upper(), where, heading_text,
                decision.threat_level))
            print(json.dumps(payload, indent=2))
        else:
            suppressed += 1
            who = decision.matched_label or classification
            print("\nsuppressed  t={0:.2f}s  {1}  at {2}{3}  ({4})".format(
                window.start_s, classification, where, heading_text, who))

    print("\n" + "=" * 70)
    print("summary: {0} alert(s) raised, {1} recognised event(s) suppressed".format(
        alerts, suppressed))
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

"""
profile_classifier.py
======================
Decision engine for the Seismic Perimeter Intelligence core.

This stage turns a conditioned activity window into an operational decision. It

  1. extracts physical features from the window (peak force, impact cadence or
     axle spacing, spectral character),
  2. classifies the source as human, vehicle or wildlife,
  3. estimates an effective mass from the peak amplitude using the linear
     seismic coupling of the site,
  4. compares the result against a set of saved profiles for known people and
     vehicles, and
  5. emits a structured JSON alert webhook when, and only when, it sees
     something unknown or something known that has changed in a way that
     matters (for example a registered vehicle that has returned carrying
     significantly more load on one axle).

Recognised, unchanged traffic is suppressed so the operator is not buried in
notifications for routine comings and goings.

The webhook schema is intentionally generic so that a Video Management System
or camera platform can consume it directly: it carries a classification, a
confidence, a threat level, the supporting kinematics and a recommended camera
response.

Depends on numpy and scipy for analysis and the Python standard library for the
alert payload. Run directly for an end-to-end demonstration:

    python profile_classifier.py
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks

from data_simulator import DEFAULT_SAMPLING_RATE_HZ, build_demo_scenario
from signal_processor import ActivityWindow, ProcessResult, SignalProcessor


# ---------------------------------------------------------------------------
# Site calibration
# ---------------------------------------------------------------------------
#
# These are end-to-end calibration constants: processed volts of envelope peak
# per kilogram of effective load, measured by walking a known mass and driving
# a known axle load across the array and reading the output of the conditioning
# chain. They are smaller than the raw physical coupling because the bandpass
# and envelope stages have a finite passband gain. In a field deployment they
# live in a per-site calibration file and are refreshed during commissioning.

PEDESTRIAN_V_PER_KG: float = 7.2e-4
AXLE_V_PER_KG: float = 8.3e-5


# ---------------------------------------------------------------------------
# Sensor node identity (would come from device configuration in the field)
# ---------------------------------------------------------------------------

SENSOR_NODE: Dict[str, object] = {
    "node_id": "GP-NODE-01",
    "array": "PERIMETER-WEST",
    "channels": 4,
    "latitude": 0.0,
    "longitude": 0.0,
    "linked_camera": "CAM-PERIM-04",
}


# ---------------------------------------------------------------------------
# Saved profiles of recognised people and assets
# ---------------------------------------------------------------------------
#
# A real system learns these from labelled passes during enrolment. The
# tolerances define how far a live reading may drift from the stored profile
# before it stops being "the same" person or vehicle.

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

    peak_v: float                 # tallest envelope value in the window
    heaviest_impact_v: float      # tallest individual impact (footfall or axle)
    num_impacts: int              # footfalls or axles detected
    impact_times_s: List[float]   # absolute time of each impact
    median_interval_s: float      # typical spacing between impacts
    interval_cv: float            # spacing variability (0 = perfectly regular)
    cadence_hz: float             # impacts per second (0 if undefined)
    dominant_freq_hz: float       # strongest in-band frequency
    low_freq_ratio: float         # share of band energy below 18 Hz
    heaviest_impact_index: int    # which impact was tallest (0-based)


def _spectral_features(segment: np.ndarray, fs: int,
                       band: tuple = (10.0, 45.0),
                       low_split_hz: float = 18.0) -> tuple:
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
    """Recover footfall or axle structure and spectral character from a window."""
    s, e = window.start_idx, window.end_idx
    env = result.envelope[s:e]
    seg = result.filtered[s:e]
    peak_v = float(env.max()) if env.size else 0.0

    # Individual impacts: footfalls for a walker, axle deflections for a vehicle.
    # Height and prominence gates keep low engine rumble from registering as an
    # axle, and the distance gate stops one impact being counted twice.
    peaks, props = find_peaks(
        env, height=0.35 * peak_v,
        distance=max(1, int(round(0.18 * fs))),
        prominence=0.2 * peak_v)
    heights = props["peak_heights"] if peaks.size else np.array([peak_v])

    impact_times = (s + peaks) / fs
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
        num_impacts=int(peaks.size) if peaks.size else 1,
        impact_times_s=[round(float(t), 3) for t in impact_times],
        median_interval_s=median_interval,
        interval_cv=interval_cv,
        cadence_hz=cadence_hz,
        dominant_freq_hz=dominant_freq,
        low_freq_ratio=low_ratio,
        heaviest_impact_index=int(np.argmax(heights)))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(features: Features) -> tuple:
    """Return (class, confidence) for a feature set.

    The rules follow the physics. A vehicle couples most of its energy into the
    low part of the band through heavy, slow axle deflections, so it is
    low-frequency dominated. A walker produces sharper, higher-frequency
    footfalls at a steady cadence. Anything weak and irregular is treated as
    wildlife or environmental and is not actionable.
    """
    dom = features.dominant_freq_hz
    low_ratio = features.low_freq_ratio
    cadence = features.cadence_hz
    peak = features.peak_v

    # Vehicle: low-frequency dominated and energetic.
    if dom < 20.0 and low_ratio > 0.45 and peak > 0.03:
        confidence = min(0.99, 0.65 + 0.34 * low_ratio)
        return "vehicle", round(confidence, 3)

    # Human: footfall band with a plausible walking cadence.
    if 20.0 <= dom <= 42.0 and 1.2 <= cadence <= 3.0 and peak > 0.02:
        regularity = max(0.0, 1.0 - min(features.interval_cv, 1.0))
        confidence = min(0.99, 0.60 + 0.38 * regularity)
        return "human", round(confidence, 3)

    # Everything else: small or irregular, not a perimeter threat.
    return "wildlife", 0.5


# ---------------------------------------------------------------------------
# Mass estimation
# ---------------------------------------------------------------------------

def estimate_mass(classification: str, features: Features) -> Dict[str, float]:
    """Convert peak amplitudes into effective masses via linear coupling."""
    if classification == "vehicle":
        heaviest_axle = features.heaviest_impact_v / AXLE_V_PER_KG
        total = sum(
            h / AXLE_V_PER_KG for h in [features.heaviest_impact_v]) if features.num_impacts == 1 \
            else features.num_impacts * (features.peak_v / AXLE_V_PER_KG)
        return {"heaviest_axle_kg": round(heaviest_axle, 1),
                "gross_estimate_kg": round(total, 1)}
    if classification == "human":
        return {"body_mass_kg": round(features.peak_v / PEDESTRIAN_V_PER_KG, 1)}
    return {"effective_kg": round(features.peak_v / PEDESTRIAN_V_PER_KG, 1)}


# ---------------------------------------------------------------------------
# Profile matching and decision
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """Operational verdict for one activity window."""

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
        cadence_ok = _within(features.cadence_hz, profile["cadence_hz"], tol["cadence_pct"])
        if not cadence_ok:
            continue  # different gait, not this person
        # Same gait signature. Has the carried mass changed materially?
        if _within(body_mass, profile["body_mass_kg"], tol["mass_pct"]):
            return Decision(
                status="authorized", threat_level="INFO", emit_alert=False,
                matched_profile_id=pid, matched_label=profile["label"],
                reasons=["gait and mass match enrolled profile"])
        delta = (body_mass - profile["body_mass_kg"]) / profile["body_mass_kg"]
        return Decision(
            status="anomalous", threat_level="ELEVATED", emit_alert=True,
            matched_profile_id=pid, matched_label=profile["label"],
            reasons=["gait matches enrolled profile but body mass deviates "
                     "by {0:+.0%}, consistent with a carried load".format(delta)],
            deviations={"body_mass_pct": round(delta, 3),
                        "expected_kg": profile["body_mass_kg"],
                        "measured_kg": body_mass})
    # No enrolled gait matched.
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
            continue  # different chassis geometry, not this vehicle
        # Same vehicle geometry. Check the load on the heaviest axle.
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
    # No registered vehicle matched the geometry.
    return Decision(
        status="unknown", threat_level="HIGH", emit_alert=True,
        reasons=["axle count or spacing does not match any registered vehicle"])


def evaluate(classification: str, features: Features,
             mass: Dict[str, float]) -> Decision:
    """Match a classified event to saved profiles and decide what to do."""
    if classification == "human":
        return _evaluate_human(features, mass)
    if classification == "vehicle":
        return _evaluate_vehicle(features, mass)
    return Decision(status="ignored", threat_level="LOW", emit_alert=False,
                    reasons=["sub-threshold or irregular, classified as wildlife"])


# ---------------------------------------------------------------------------
# Alert payload
# ---------------------------------------------------------------------------

def build_alert_payload(window: ActivityWindow, classification: str,
                        confidence: float, features: Features,
                        mass: Dict[str, float], decision: Decision) -> Dict:
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
        wheelbase = SAVED_PROFILES.get(decision.matched_profile_id or "", {}).get("wheelbase_m")
        if wheelbase and features.median_interval_s > 0:
            kinematics["estimated_speed_m_s"] = round(wheelbase / features.median_interval_s, 2)

    # Map the verdict onto a concrete camera action the VMS can execute.
    dispatch = decision.threat_level in ("ELEVATED", "HIGH")
    response = {
        "dispatch": dispatch,
        "vms_action": "GOTO_PRESET" if dispatch else "LOG_ONLY",
        "camera_id": SENSOR_NODE["linked_camera"],
        "ptz_preset": 7 if dispatch else None,
        "notify_channels": ["soc-webhook", "guard-mobile"] if dispatch else [],
    }

    return {
        "schema": "seismic.alert.v1",
        "event_id": str(uuid.uuid4()),
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "sensor": dict(SENSOR_NODE),
        "detection": {
            "window": {
                "start_s": round(window.start_s, 2),
                "end_s": round(window.end_s, 2),
                "duration_s": round(window.duration_s, 2),
            },
            "classification": classification,
            "confidence": confidence,
            "peak_amplitude_v": round(features.peak_v, 4),
        },
        "kinematics": kinematics,
        "mass_estimate": {"method": "linear_seismic_coupling", **mass},
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
    """Simulate, condition, classify and decide over the reference scenario."""
    raw = build_demo_scenario(fs=fs, seed=seed)
    result = SignalProcessor(fs=fs).process(raw)

    print("=" * 70)
    print("SEISMIC PERIMETER INTELLIGENCE  |  live decision stream")
    print("node {0} / array {1} / {2} detected activity windows".format(
        SENSOR_NODE["node_id"], SENSOR_NODE["array"], len(result.windows)))
    print("=" * 70)

    alerts = 0
    suppressed = 0
    for window in result.windows:
        features = extract_features(result, window, fs)
        classification, confidence = classify(features)
        mass = estimate_mass(classification, features)
        decision = evaluate(classification, features, mass)

        if decision.emit_alert:
            alerts += 1
            payload = build_alert_payload(window, classification, confidence,
                                          features, mass, decision)
            print("\nALERT  t={0:.2f}s  {1}  [{2}]".format(
                window.start_s, classification.upper(), decision.threat_level))
            print(json.dumps(payload, indent=2))
        else:
            suppressed += 1
            who = decision.matched_label or classification
            print("\nsuppressed  t={0:.2f}s  {1}  ({2})".format(
                window.start_s, classification, who))

    print("\n" + "=" * 70)
    print("summary: {0} alert(s) raised, {1} recognised event(s) suppressed".format(
        alerts, suppressed))
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()

# Seismic Perimeter Intelligence Core

A physics-informed software stack that turns buried geophones (passive ground
vibration sensors) into a smart perimeter security and asset-tracking layer. It
classifies what is moving across a site from seismic waves alone, recognises the
people and vehicles that belong there, and raises a structured alert the moment
it sees something unknown or something known that has changed in a way that
matters.

The processing and decision logic in this repository is the deployable product.
A built-in simulation harness stands in for the live sensor feed so the entire
stack can be run, tested and reviewed end to end today, and so the same code
path drops straight onto hardware once the array is wired up.

## Why seismic sensing

Buried geophones give a perimeter layer that cameras and fences cannot:

- Passive and covert. There is nothing above ground to see, avoid or disable.
- No line of sight required. The sensor responds to ground motion, so foliage,
  darkness, fog and weather do not blind it.
- All-weather and low power. A geophone is a coil and a magnet; the
  intelligence is entirely in software.
- Inherently physical. The signal carries force and mass information, not just
  presence, which is what enables payload-aware tracking.

## What it does

1. Classifies a moving target as human, vehicle or wildlife from its seismic
   signature.
2. Matches the signature against saved profiles of enrolled owners and
   registered vehicles, and suppresses routine, recognised traffic.
3. Fires a JSON alert webhook for any unknown target, or for a known target
   whose physical profile has shifted, for example a registered vehicle that
   returns with significantly more load on one axle.

## The physics: linear mass to amplitude coupling

A geophone outputs a voltage proportional to ground velocity. The peak
amplitude of the disturbance produced by a footfall or a wheel axle is, to first
order, linear in the dynamic load that caused it:

```
peak_amplitude  =  coupling_constant  x  effective_mass
```

That single linear relationship is the backbone of the system. Because
amplitude scales with mass, the conditioning chain can be calibrated once,
during commissioning, by walking a known mass and driving a known axle load
across the array. From then on the inverse mapping recovers an effective mass
from any measured amplitude. A registered pickup whose rear axle reads twenty
percent heavier than its enrolled baseline is therefore not a guess; it is a
direct, calibrated measurement. The model scales linearly and predictably as
load changes, which keeps the thresholds interpretable and the false-alarm
behaviour stable.

## Architecture

```
   raw geophone voltage (100 Hz)
              |
   +----------v-----------+      data_simulator.py
   |  acquisition / sim    |      synthetic feed with a known ground truth
   +----------+-----------+
              |
   +----------v-----------+      signal_processor.py
   |  bandpass 10-50 Hz    |      isolate the footfall / axle band
   |  adaptive baseline    |      rolling-median threshold tracks the noise floor
   |  event detection      |      group activity into windows
   +----------+-----------+
              |
   +----------v-----------+      profile_classifier.py
   |  feature extraction   |      peak force, cadence / axle spacing, spectrum
   |  classification       |      human / vehicle / wildlife
   |  mass estimation      |      linear seismic coupling
   |  profile matching     |      compare against enrolled owners and vehicles
   +----------+-----------+
              |
   +----------v-----------+
   |  JSON alert webhook    |      consumed by the VMS / camera platform
   +----------------------+
```

## Modules

| File                    | Responsibility |
| ----------------------- | -------------- |
| `data_simulator.py`     | Generates synthetic 100 Hz geophone feeds with a known ground truth: a flat noise floor plus injectable human footstep trains, overloaded footstep trains and multi-axle vehicle passes with engine rumble. |
| `signal_processor.py`   | Zero-phase Butterworth bandpass (10 to 50 Hz) and an adaptive rolling-median threshold that tracks slow environmental drift, then groups threshold crossings into activity windows. |
| `profile_classifier.py` | Extracts features, classifies the source, estimates effective mass from amplitude, matches against saved profiles and emits the JSON alert webhook. |

## Quickstart

```bash
git clone <your-repo-url>
cd geophone
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline against the reference scenario and watch alerts stream
python profile_classifier.py
```

Each module is also runnable on its own:

```bash
python data_simulator.py            # generate a feed and print a summary
python data_simulator.py --csv feed.csv --duration 30   # export samples
python signal_processor.py          # show the detected activity windows
```

## The reference scenario

`python profile_classifier.py` runs a deterministic 30 second timeline:

| Time | Event | Expected verdict |
| ---- | ----- | ---------------- |
| 3 s  | Property owner on foot (75 kg, cadence 1.8 Hz) | recognised, suppressed |
| 10 s | Unidentified person on foot (heavier, faster)  | unknown, HIGH alert |
| 16 s | Owner pickup, unladen (two axles)              | recognised, suppressed |
| 23 s | Owner pickup returning, rear axle loaded       | anomalous payload, ELEVATED alert |

Two recognised events are silently suppressed and two alerts are raised: one for
the unknown pedestrian and one for the registered vehicle returning with extra
load on its trailing axle.

## Example alert

```json
{
  "schema": "seismic.alert.v1",
  "event_id": "79cff9c5-db84-4620-beb2-ce56a5bc0fd0",
  "emitted_at": "2026-06-17T02:24:52+00:00",
  "sensor": { "node_id": "GP-NODE-01", "array": "PERIMETER-WEST", "channels": 4 },
  "detection": { "classification": "vehicle", "confidence": 0.956, "peak_amplitude_v": 0.0631 },
  "kinematics": { "impacts": 2, "axle_spacing_s": 0.45, "estimated_speed_m_s": 7.11, "low_frequency_ratio": 0.9 },
  "mass_estimate": { "method": "linear_seismic_coupling", "heaviest_axle_kg": 760.6, "gross_estimate_kg": 1521.2 },
  "identity": {
    "status": "anomalous",
    "matched_profile": "OWN-VEH-01",
    "deviations": { "axle_load_pct": 0.17, "axle_position": "trailing", "expected_kg": 650.0, "measured_kg": 760.6 }
  },
  "threat": { "level": "ELEVATED", "reasons": ["registered vehicle returned with +17% load change on the trailing axle, consistent with added payload"] },
  "response": { "dispatch": true, "vms_action": "GOTO_PRESET", "camera_id": "CAM-PERIM-04", "ptz_preset": 7 }
}
```

## Adaptive baseline

Fixed trigger levels fail outdoors because the noise floor moves with wind,
rain and temperature. The processor instead derives its threshold from a rolling
median of the signal envelope plus a multiple of the rolling median absolute
deviation. Medians are insensitive to the brief, large excursions that genuine
events cause, so an event never inflates its own threshold, while slow changes
in the environment raise the floor gradually and silently. The result is high
sensitivity to real activity and stable false-alarm behaviour across conditions.

## Integration with Video Management Systems

The alert is a plain JSON document over a webhook, which is the integration
contract every modern VMS already understands. The `response` block names a
concrete camera and a PTZ preset so that a seismic detection can cue a camera to
the right spot before the target is in frame, turning a non-visual sensor into a
trigger for the existing video estate. A small bridge maps the webhook onto a
specific platform (for example an ONVIF event or a vendor REST endpoint) without
any change to the core. Because recognised traffic is suppressed upstream, the
VMS only receives events that warrant a look.

## Reference hardware target

The intended field node is deliberately low cost and off the shelf:

- Raspberry Pi class single-board computer
- ADS1115 16-bit analog-to-digital converter
- Four geophone elements for an array, giving direction and localisation headroom

The acquisition interface is isolated behind `data_simulator.py` today; swapping
the simulated feed for the live ADC stream leaves the processing and
classification stages unchanged.

## Roadmap

- Multi-channel array processing for bearing and range estimation
- Learned classifiers trained on labelled field captures, replacing the
  rule-based decision layer while keeping the same physics-derived features
- On-device enrolment workflow for owners and vehicles
- Direction-of-travel and trajectory tracking across multiple nodes

## Repository layout

```
geophone/
  data_simulator.py      synthetic geophone feed generator
  signal_processor.py    filtering, adaptive baseline, event detection
  profile_classifier.py  features, classification, profiles, JSON alerts
  requirements.txt       numpy, scipy
  README.md              this document
```

## License

Released under the MIT License. See `LICENSE` if present, or add one before
publishing.

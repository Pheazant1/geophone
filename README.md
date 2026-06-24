# Seismic Perimeter Intelligence Core

A physics-informed software stack that turns a buried four-element geophone
array (passive ground vibration sensors) into a smart perimeter security and
asset-tracking layer. It classifies what is moving across a site from seismic
waves alone, works out where it is and which way it is heading, recognises the
people and vehicles that belong there, and raises a structured alert the moment
it sees something unknown or something known that has changed in a way that
matters.

The processing, localisation and decision logic in this repository is the
deployable product. A built-in simulation harness stands in for the live sensor
feed so the entire stack can be run, tested and reviewed end to end today, and
so the same code path drops straight onto hardware once the array is wired up.

## Why seismic sensing

A buried geophone array gives a perimeter layer that cameras and fences cannot:

- Passive and covert. There is nothing above ground to see, avoid or disable.
- No line of sight required. The sensor responds to ground motion, so foliage,
  darkness, fog and weather do not blind it.
- All-weather and low power. A geophone is a coil and a magnet; the
  intelligence is entirely in software.
- Inherently physical. The signal carries force, mass and location information,
  not just presence, which is what enables payload-aware tracking and the
  cueing of cameras to an exact spot.

## What it does

1. Classifies a moving target as human, vehicle or wildlife from its seismic
   signature.
2. Localises the target inside the array footprint and, for a moving target,
   reports a heading and speed.
3. Estimates an effective mass from the peak amplitude, corrected for the range
   to the sensor, using the linear seismic coupling of the site.
4. Matches the signature against saved profiles of enrolled owners and
   registered vehicles, and suppresses routine, recognised traffic.
5. Fires a JSON alert webhook for any unknown target, or for a known target
   whose physical profile has shifted, for example a registered vehicle that
   returns with significantly more load on one axle. The alert names the camera
   that covers the detected location.

## The physics

Two linear relationships do the heavy lifting.

Mass to amplitude. A geophone outputs a voltage proportional to ground
velocity, and the peak amplitude of a footfall or an axle deflection is, to
first order, linear in the dynamic load that caused it:

```
reference_amplitude  =  coupling_constant  x  effective_mass
```

Distance to amplitude. As the surface wave travels it spreads out and the
ground absorbs energy, so the amplitude falls off in a known way with range.
Once the array has localised the source, the measured amplitude is corrected
back to a reference distance before the mass is read off.

Together these mean a registered pickup whose rear axle reads twenty percent
heavier than its enrolled baseline is not a guess; it is a calibrated
measurement, taken after accounting for exactly where the vehicle was. The model
scales linearly and predictably as load changes, which keeps the thresholds
interpretable and the false-alarm behaviour stable.

Location comes from timing. The same disturbance reaches the four sensors at
slightly different times. Those time differences of arrival, measured by
cross-correlation, pin down the source: travel time equals distance over wave
speed, and three independent time differences fix the position inside the array.

## Architecture

```
   four geophone channels (100 Hz)
              |
   +----------v-----------+      data_simulator.py
   |  acquisition / sim    |      synthetic array feed with a known ground truth
   +----------+-----------+
              |
   +----------v-----------+      signal_processor.py
   |  bandpass 10-50 Hz    |      isolate the footfall / axle band per channel
   |  adaptive baseline    |      rolling-median threshold tracks the noise floor
   |  event detection      |      group activity into windows across the array
   +----------+-----------+
              |
        +-----+------+----------------------+
        |            |                       |
   +----v----+  +----v---------+      +------v-----------+
   | localizer|  | feature      |      | (per window)     |
   | TDOA +   |  | extraction   |      |                  |
   | position |  | cadence /    |      |                  |
   | heading  |  | axle / spectrum|    |                  |
   +----+----+  +----+---------+      +------------------+
        |            |
   +----v------------v----+      profile_classifier.py
   |  classification       |      human / vehicle / wildlife
   |  range-corrected mass |      linear coupling + distance correction
   |  profile matching     |      compare against enrolled owners and vehicles
   +----------+-----------+
              |
   +----------v-----------+
   |  JSON alert webhook    |      classification, location, mass, camera cue
   +----------------------+
```

## Modules

| File                    | Responsibility |
| ----------------------- | -------------- |
| `data_simulator.py`     | Generates synthetic 100 Hz feeds for a four-element array with a known ground truth: a noise floor plus injectable footstep trains and vehicle passes that travel along real paths, each impact reaching every sensor with the correct delay and distance attenuation. |
| `signal_processor.py`   | Zero-phase Butterworth bandpass (10 to 50 Hz) and an adaptive rolling-median threshold per channel, then groups threshold crossings on the array-averaged envelope into activity windows and tags each with the strongest channel. |
| `localizer.py`          | Measures time differences of arrival by cross-correlation, solves for the source position by grid search, and resolves a heading and speed for events that move far enough. |
| `profile_classifier.py` | Extracts features, classifies the source, range-corrects mass, matches against saved profiles, picks the covering camera and emits the JSON alert webhook. |
| `pattern_memory.py`     | The self-learning core: an unsupervised pattern memory that clusters recurring feature vectors into profiles, confirms the ones that repeat, and flags anything novel. Carries no labels and no built-in idea of what a human or vehicle is. |
| `v1_demo.py`            | Wires the physics stack into `pattern_memory.py` and runs the label-free learning demonstration: a site is learned from scratch, then guarded, recognising its regulars and alerting on unseen signatures. |

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
python data_simulator.py            # generate an array feed and print a summary
python data_simulator.py --csv feed.csv --duration 30   # export per-channel samples
python signal_processor.py          # show the detected activity windows
```

## The reference scenario

`python profile_classifier.py` runs a deterministic 30 second timeline on a
30 metre square array with sensors at the corners:

| Time | Event | Expected verdict |
| ---- | ----- | ---------------- |
| 3 s  | Property owner on foot, walking north-east (75 kg, cadence 1.8 Hz) | recognised, suppressed |
| 10 s | Unidentified person on foot, walking north-west (heavier, faster)  | unknown, HIGH alert |
| 16 s | Owner pickup at the gate, unladen (two axles)                      | recognised, suppressed |
| 23 s | Owner pickup at the same gate, rear axle loaded                    | anomalous payload, ELEVATED alert |

The system suppresses the two recognised events and raises two alerts: one for
the unknown pedestrian, localised in the south-east with a north-west heading,
and one for the registered vehicle returning to the gate with extra load on its
trailing axle. Each alert names the camera covering its location.

## Self-learning mode (label-free)

The rule-based decision engine above knows in advance what a human and a vehicle
look like. The self-learning mode does the opposite: it starts from a blank
slate, with no labels and no training data, and learns the regulars at a site
the way a music-identification app learns songs. It is a general tool for
research, conservation and monitoring, where the question is not "is this
allowed" but "have I seen this before".

Each detected event is reduced to a compact, label-free signature: a
range-corrected amplitude (how large the source is), the dominant frequency and
low-frequency ratio (its contact texture), and a robust cadence recovered from
the envelope modulation spectrum (its gait or axle rhythm). The pattern memory
in `pattern_memory.py` then does three things, with nothing but running means
and a distance measure whose per-feature scale is the natural repeatability of
each feature:

- Remembers new signatures as tentative profiles.
- Confirms the ones that recur. A profile seen enough times is enrolled; a
  one-off oddity stays tentative and is treated as noise.
- Recognises and flags. Once trained, an event matching an enrolled signature
  is recognised; one that matches nothing is novel and worth attention.

```bash
python v1_demo.py
```

The demonstration runs two phases against one growing memory. In the learning
phase a stream of routine traffic arrives, three recurring sources plus a
one-off animal, and the memory clusters them with no prior knowledge into three
enrolled signatures, leaving the animal tentative. In the guarding phase the
three regulars return and are recognised and suppressed, while two never-seen
sources, an unfamiliar walker and an unfamiliar vehicle, are each flagged as a
novel signature. The run prints the distance behind every decision and, for the
demonstration only, checks each learned cluster against the simulator ground
truth to confirm that one cluster corresponds to one real source.

There is no neural network and no labelled data anywhere in this mode: the
learning is interpretable statistical pattern memory, which is what lets it run
on the same low-cost edge hardware and have every match be explainable.

## Example alert

```json
{
  "schema": "seismic.alert.v1",
  "event_id": "30f015b0-7ce9-42fb-a9ba-df75da710943",
  "emitted_at": "2026-06-17T02:56:20+00:00",
  "sensor": { "node_id": "GP-NODE-01", "array": "PERIMETER-WEST", "channels": 4 },
  "detection": { "classification": "vehicle", "confidence": 0.961, "strongest_channel": 3, "peak_amplitude_v": 0.0325 },
  "localization": { "x_m": 15.0, "y_m": 22.5, "range_m": 16.8, "residual_s": 0.0009 },
  "kinematics": { "impacts": 2, "axle_spacing_s": 0.48, "low_frequency_ratio": 0.915 },
  "mass_estimate": { "method": "range_corrected_linear_coupling", "heaviest_axle_kg": 788.8, "gross_estimate_kg": 1420.3 },
  "identity": {
    "status": "anomalous",
    "matched_profile": "OWN-VEH-01",
    "deviations": { "axle_load_pct": 0.214, "axle_position": "trailing", "expected_kg": 650.0, "measured_kg": 788.8 }
  },
  "threat": { "level": "ELEVATED", "reasons": ["registered vehicle returned with +21% load change on the trailing axle, consistent with added payload"] },
  "response": { "dispatch": true, "vms_action": "GOTO_PRESET", "camera_id": "CAM-NE-01", "ptz_preset": 1 }
}
```

## Adaptive baseline

Fixed trigger levels fail outdoors because the noise floor moves with wind, rain
and temperature. The processor instead derives its threshold from a rolling
median of the signal envelope plus a multiple of the rolling median absolute
deviation. Medians are insensitive to the brief, large excursions that genuine
events cause, so an event never inflates its own threshold, while slow changes
in the environment raise the floor gradually and silently. The result is high
sensitivity to real activity and stable false-alarm behaviour across conditions.

## Integration with Video Management Systems

The alert is a plain JSON document over a webhook, which is the integration
contract every modern VMS already understands. Because the array localises the
source, the `response` block names a specific camera and PTZ preset for the
quadrant the target is in, so a seismic detection can cue the right camera to
the right spot before the target is in frame. A small bridge maps the webhook
onto a particular platform (for example an ONVIF event or a vendor REST
endpoint) without any change to the core. Recognised traffic is suppressed
upstream, so the VMS only receives events that warrant a look.

## Reference hardware target

The intended field node is deliberately low cost and off the shelf:

- Raspberry Pi class single-board computer
- ADS1115 16-bit analog-to-digital converter
- Four geophone elements forming the array

The acquisition interface is isolated behind `data_simulator.py` today; swapping
the simulated feed for the live ADC stream leaves the processing, localisation
and classification stages unchanged. Site commissioning measures the local wave
speed and the amplitude calibration by walking a known mass and driving a known
axle load across the array.

## Roadmap

- Learned classifiers trained on labelled field captures, replacing the
  rule-based decision layer while keeping the same physics-derived features
- Multi-node fusion for tracking across a larger site
- On-device enrolment workflow for owners and vehicles
- Sensor health monitoring and automatic recalibration

## Repository layout

```
geophone/
  data_simulator.py      synthetic four-channel array feed generator
  signal_processor.py    filtering, adaptive baseline, event detection
  localizer.py           TDOA localisation, heading and speed
  profile_classifier.py  features, classification, profiles, JSON alerts
  pattern_memory.py      unsupervised self-learning pattern memory
  v1_demo.py             label-free learning and guarding demonstration
  requirements.txt       numpy, scipy
  README.md              this document
```

## License

Released under the MIT License. See `LICENSE` if present, or add one before
publishing.

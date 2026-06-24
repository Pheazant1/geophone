"""
stage1_simulation.py
=====================
Stage 1 of the continuous-learning trial: a blind rediscovery test.

A fixed, secret cast of characters goes about a weekly routine across a
simulated fortnight. Each time one of them crosses the sensor footprint it makes
a seismic event, which is fed to the self-learning pattern memory from
``pattern_memory.py``. The memory never sees the cast list or the schedule; it
sees only vibration. The question Stage 1 answers is simple and measurable: left
to itself, does the system independently rediscover the cast, building one
profile per real character?

This is deliberately the clean case. The cast is fixed, the routines are mostly
scheduled, and there is no weather drift, no mailman, no one-off wildlife and no
two sources crossing at the same instant. Those belong to Stage 2. Stage 1
exists to prove the learning and the scoring harness work end to end before any
of that is added.

How time is modelled
--------------------
A real 24/7 feed is almost entirely silence, so rather than synthesise every
quiet second this advances a clock day by day and, at each scheduled crossing,
synthesises a short clip containing the background noise plus that character's
signature. The clip runs through the real pipeline (filtering, detection,
localisation, fingerprinting) and the resulting event updates the memory, in
chronological order, exactly as a live deployment would see them. Departures and
returns are separate crossings of the same person, so they reinforce the same
profile.

Run it:

    python stage1_simulation.py
    python stage1_simulation.py --days 14 --seed 1
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from data_simulator import DEFAULT_SAMPLING_RATE_HZ, GeophoneArraySimulator
from pattern_memory import PatternMemory
from signal_processor import SignalProcessor
from v1_demo import FEATURE_NAMES, FEATURE_SCALES, build_feature_vector


WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WORK_DAYS: Set[int] = {0, 1, 2, 3, 4}
EVERY_DAY: Set[int] = {0, 1, 2, 3, 4, 5, 6}


# ---------------------------------------------------------------------------
# The secret cast (ground truth; the memory never sees any of this)
# ---------------------------------------------------------------------------

@dataclass
class Routine:
    """When a character crosses the array: times of day, on which weekdays."""

    crossings: List[Tuple[int, int]]   # list of (hour, minute)
    weekdays: Set[int]                  # 0 = Monday .. 6 = Sunday


@dataclass
class Character:
    """A recurring source with stable traits and a weekly routine.

    Traits carry small natural variation on every appearance, so the memory has
    to recognise a source that is similar but never identical between visits,
    which is the real test of the matcher.
    """

    name: str
    kind: str                           # "human" | "animal" | "vehicle"
    routine: Routine
    path_start: Tuple[float, float]
    path_end: Tuple[float, float]
    # locomotion (human and animal)
    mass_kg: float = 0.0
    cadence_hz: float = 0.0
    step_freq_hz: float = 30.0
    num_steps: int = 6
    # vehicle
    axle_masses: Tuple[float, ...] = (600.0, 600.0)
    axle_spacing_s: float = 0.45

    def emit(self, sim: GeophoneArraySimulator, feed: np.ndarray,
             onset_s: float, rng: np.random.Generator) -> None:
        """Inject one appearance of this character into a clip, with variation."""
        start = np.asarray(self.path_start, dtype=float) + rng.normal(0.0, 1.0, 2)
        end = np.asarray(self.path_end, dtype=float) + rng.normal(0.0, 1.0, 2)
        if self.kind in ("human", "animal"):
            sim.add_footstep_train(
                feed, start_s=onset_s, path_start_xy=start, path_end_xy=end,
                num_steps=self.num_steps,
                cadence_hz=float(rng.normal(self.cadence_hz, 0.05)),
                mass_kg=float(rng.normal(self.mass_kg, 0.03 * self.mass_kg)),
                freq_hz=self.step_freq_hz, label=self.name)
        else:
            axles = tuple(float(rng.normal(m, 0.02 * m)) for m in self.axle_masses)
            sim.add_vehicle_pass(
                feed, start_s=onset_s, path_start_xy=start, path_end_xy=end,
                axle_masses=axles, axle_spacing_s=self.axle_spacing_s,
                label=self.name)


def build_cast() -> List[Character]:
    """The fixed Stage 1 cast: three people, a car and a dog.

    The two adults are about the same weight on purpose; they are told apart by
    pace, not size. The child is lighter and quicker. The car and dog are
    seismically very different from the walkers and from each other.
    """
    return [
        Character("Adult-1 (approx 80 kg, slow walker)", "human",
                  Routine([(8, 0), (17, 0)], WORK_DAYS),
                  path_start=(8.0, 10.0), path_end=(14.0, 16.0),
                  mass_kg=80.0, cadence_hz=1.6, step_freq_hz=30.0, num_steps=6),
        Character("Adult-2 (approx 78 kg, brisk walker)", "human",
                  Routine([(8, 30), (17, 30)], WORK_DAYS),
                  path_start=(22.0, 9.0), path_end=(17.0, 15.0),
                  mass_kg=78.0, cadence_hz=2.2, step_freq_hz=32.0, num_steps=6),
        Character("Child (approx 35 kg)", "human",
                  Routine([(8, 15), (15, 30)], WORK_DAYS),
                  path_start=(10.0, 20.0), path_end=(16.0, 24.0),
                  mass_kg=35.0, cadence_hz=2.5, step_freq_hz=36.0, num_steps=6),
        Character("Car", "vehicle",
                  Routine([(7, 50), (18, 0)], WORK_DAYS),
                  path_start=(15.0, 22.0), path_end=(15.0, 22.0),
                  axle_masses=(620.0, 640.0), axle_spacing_s=0.45),
        Character("Dog (approx 45 kg)", "animal",
                  Routine([(7, 30), (12, 30), (19, 0)], EVERY_DAY),
                  path_start=(11.0, 11.0), path_end=(18.0, 15.0),
                  mass_kg=45.0, cadence_hz=3.0, step_freq_hz=40.0, num_steps=8),
    ]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

@dataclass
class ScheduledEvent:
    day: int
    weekday: int
    minute_of_day: int
    character: Character


def build_timeline(cast: Sequence[Character], num_days: int,
                   rng: np.random.Generator,
                   punctuality_min: int = 15) -> List[ScheduledEvent]:
    """Expand the weekly routines into a chronological list of crossings.

    Each crossing time is jittered by up to ``punctuality_min`` minutes so the
    cast is realistically imprecise rather than perfectly clockwork.
    """
    events: List[ScheduledEvent] = []
    for day in range(num_days):
        weekday = day % 7
        for character in cast:
            if weekday not in character.routine.weekdays:
                continue
            for hour, minute in character.routine.crossings:
                jitter = int(rng.integers(-punctuality_min, punctuality_min + 1))
                minute_of_day = hour * 60 + minute + jitter
                events.append(ScheduledEvent(day, weekday, minute_of_day, character))
    events.sort(key=lambda e: (e.day, e.minute_of_day))
    return events


def _clock(minute_of_day: int) -> str:
    minute_of_day = max(0, min(minute_of_day, 24 * 60 - 1))
    return "{0:02d}:{1:02d}".format(minute_of_day // 60, minute_of_day % 60)


# ---------------------------------------------------------------------------
# The trial
# ---------------------------------------------------------------------------

def _print_plan(cast: Sequence[Character], num_days: int) -> None:
    print("SECRET CAST AND ROUTINE  (the system is never told any of this)\n")
    for character in cast:
        days = ("Mon-Fri" if character.routine.weekdays == WORK_DAYS
                else "every day" if character.routine.weekdays == EVERY_DAY
                else ",".join(WEEKDAY_NAMES[d] for d in sorted(character.routine.weekdays)))
        times = ", ".join(_clock(h * 60 + m) for h, m in character.routine.crossings)
        print("  {0:<38}  {1:<9}  crosses at {2}".format(
            character.name, days, times))
    print("\n  simulating {0} days, then comparing what the system learned "
          "to this list".format(num_days))


def run_trial(num_days: int = 14, seed: int = 1,
              fs: int = DEFAULT_SAMPLING_RATE_HZ,
              match_distance: float = 1.2, enroll_after: int = 3,
              verbose: bool = True) -> None:
    cast = build_cast()
    rng = np.random.default_rng(seed)
    sim = GeophoneArraySimulator(sampling_rate_hz=fs, seed=seed)
    processor = SignalProcessor(fs=fs, merge_gap_s=1.2)
    memory = PatternMemory(FEATURE_SCALES, match_distance=match_distance,
                           enroll_after=enroll_after, feature_names=FEATURE_NAMES)

    bar = "=" * 74
    print(bar)
    print("STAGE 1  |  blind rediscovery trial  |  self-learning seismic memory")
    print(bar + "\n")
    _print_plan(cast, num_days)
    print("\n" + bar)
    print("ACCELERATED CLOCK  (each line is one crossing the system detected)\n")

    timeline = build_timeline(cast, num_days, rng)
    # profile id -> how many crossings of each true character landed in it
    profile_truth: Dict[str, Counter] = {}
    detections = 0
    misses = 0
    current_day = -1

    for event in timeline:
        if event.day != current_day:
            current_day = event.day
            print("\n-- Day {0:02d} ({1}) --".format(
                event.day + 1, WEEKDAY_NAMES[event.weekday]))

        feed = sim.baseline(duration_s=12.0)
        event.character.emit(sim, feed, onset_s=3.0, rng=rng)
        result = processor.process(feed)
        if not result.windows:
            misses += 1
            if verbose:
                print("   {0}  {1:<38}  (too faint to detect)".format(
                    _clock(event.minute_of_day), event.character.name))
            continue

        window = max(result.windows, key=lambda w: w.peak_v)
        vector, _, readable = build_feature_vector(result, window, fs)
        obs = memory.observe(vector, float(event.day * 1440 + event.minute_of_day))
        detections += 1
        profile_truth.setdefault(obs.profile_id, Counter())[event.character.name] += 1

        if verbose:
            if obs.is_new:
                verdict = "NEW signature {0}".format(obs.profile_id)
            elif obs.newly_enrolled:
                verdict = "{0} CONFIRMED (now enrolled, {1} obs)".format(
                    obs.profile_id, obs.observations)
            else:
                verdict = "{0} matched (d={1:.2f}, {2} obs)".format(
                    obs.profile_id, obs.distance, obs.observations)
            print("   {0}  {1:<38}  -> {2}".format(
                _clock(event.minute_of_day), event.character.name, verdict))

    # Maintenance: repair any same-source splits before scoring.
    merges = memory.consolidate()
    for removed_id, kept_id in merges:
        if removed_id in profile_truth:
            profile_truth.setdefault(kept_id, Counter()).update(profile_truth.pop(removed_id))
    if merges:
        print("\n  (consolidation merged {0} same-source split(s))".format(len(merges)))

    _print_scorecard(cast, memory, profile_truth, detections, misses, bar)


# ---------------------------------------------------------------------------
# Scorecard: did the blind system match the secret cast?
# ---------------------------------------------------------------------------

def _print_scorecard(cast: Sequence[Character], memory: PatternMemory,
                     profile_truth: Dict[str, Counter],
                     detections: int, misses: int, bar: str) -> None:
    enrolled = memory.enrolled_profiles()
    tentative = memory.tentative_profiles()

    print("\n" + bar)
    print("SCORECARD  (system's blind result vs the secret cast)\n")
    print("  crossings detected : {0}".format(detections))
    if misses:
        print("  crossings too faint to detect : {0}".format(misses))
    print("  true characters    : {0}".format(len(cast)))
    print("  enrolled profiles  : {0}".format(len(enrolled)))
    if tentative:
        print("  tentative (noise)  : {0}".format(len(tentative)))

    print("\n  what each learned profile turned out to be:")
    profile_to_char: Dict[str, str] = {}
    for profile in enrolled:
        counts = profile_truth.get(profile.profile_id, Counter())
        total = sum(counts.values())
        if total == 0:
            continue
        dominant, dom_count = counts.most_common(1)[0]
        purity = dom_count / total
        profile_to_char[profile.profile_id] = dominant
        extra = ""
        if len(counts) > 1:
            others = ", ".join("{0}x {1}".format(c, n)
                               for n, c in counts.most_common()[1:])
            extra = "  (also caught: {0})".format(others)
        print("    {0}  ->  {1}   [{2} obs, {3:.0%} pure]{4}".format(
            profile.profile_id, dominant, total, purity, extra))

    print("\n  did every real character get its own profile:")
    char_to_profiles: Dict[str, List[str]] = {c.name: [] for c in cast}
    for pid, name in profile_to_char.items():
        char_to_profiles.setdefault(name, []).append(pid)
    for character in cast:
        pids = char_to_profiles.get(character.name, [])
        if len(pids) == 1:
            mark = "OK    "
            note = "recognised as {0}".format(pids[0])
        elif len(pids) == 0:
            mark = "MISS  "
            note = "no profile (merged into another character or never enrolled)"
        else:
            mark = "SPLIT "
            note = "split across {0}".format(", ".join(pids))
        print("    [{0}] {1:<38} {2}".format(mark, character.name, note))

    # Verdict: a clean pass is a one-to-one match of profiles and characters,
    # each profile dominated by a single character.
    one_to_one = (len(enrolled) == len(cast)
                  and all(len(char_to_profiles.get(c.name, [])) == 1 for c in cast))
    pure = all(
        (sum(profile_truth.get(p.profile_id, Counter()).values()) == 0)
        or (profile_truth[p.profile_id].most_common(1)[0][1]
            / sum(profile_truth[p.profile_id].values()) >= 0.9)
        for p in enrolled)

    print("\n" + bar)
    if one_to_one and pure:
        print("VERDICT: PASS. The system rediscovered all {0} characters blind, "
              "one profile each.".format(len(cast)))
    else:
        print("VERDICT: PARTIAL. The system learned {0} profile(s) for {1} "
              "characters; see the merges/splits above.".format(
                  len(enrolled), len(cast)))
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 blind rediscovery trial for the self-learning core.")
    parser.add_argument("--days", type=int, default=14,
                        help="number of days to simulate (default 14)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed for reproducibility (default 1)")
    parser.add_argument("--match-distance", type=float, default=1.2,
                        help="match threshold in feature-scale units (default 1.2)")
    parser.add_argument("--enroll-after", type=int, default=3,
                        help="sightings needed to enrol a signature (default 3)")
    parser.add_argument("--quiet", action="store_true",
                        help="hide the per-crossing log, show only the scorecard")
    args = parser.parse_args()
    run_trial(num_days=args.days, seed=args.seed,
              match_distance=args.match_distance, enroll_after=args.enroll_after,
              verbose=not args.quiet)


if __name__ == "__main__":
    main()

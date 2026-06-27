"""
pattern_memory.py
=================
Unsupervised, self-learning pattern memory for the open-source V1 core.

This is the "brain" of the V1 product. Unlike the V2 decision engine in
``profile_classifier.py``, it carries no built-in knowledge of what a human, a
vehicle or an animal is. It is handed a stream of feature vectors, each one a
compact fingerprint of a seismic event, and it does exactly three things:

  1. Remembers signatures it has seen. A new event that resembles nothing in
     memory becomes a tentative profile.
  2. Confirms the ones that recur. A tentative profile that is seen again and
     again, above a confirmation count, is promoted to an enrolled signature.
     Profiles that appear once and never return stay tentative and are treated
     as noise.
  3. Recognises and flags. Once trained, an incoming event that matches an
     enrolled signature is recognised; one that matches nothing is novel and is
     the thing worth raising an alarm about.

There is no training data, no labels and no neural network. The learning is
statistical pattern memory: running means of feature vectors, compared with a
distance measure whose per-feature scale is the natural measurement
repeatability of each feature. That is the right metric for identity, because
two events are the same source when their features agree to within how
repeatably each feature can be measured, not to within how much the features
vary across the whole population. It runs on a Raspberry Pi, needs no GPU and
is fully interpretable: every match comes with the distance that justified it.

This module depends only on numpy and the standard library. It is deliberately
free of any seismic or simulator import, so the same brain can be reused on any
feature stream. The seismic feature extraction that feeds it lives in
``v1_demo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# A learned signature
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    """One remembered signature, held as a running mean of its observations.

    A profile starts tentative on the first sighting and is promoted to
    enrolled once it has recurred ``enroll_after`` times. The mean is updated
    incrementally, so a profile gently averages out the natural variation
    between repeat visits of the same source rather than locking onto the first
    sample.
    """

    profile_id: str
    mean: np.ndarray
    count: int
    first_seen_s: float
    last_seen_s: float
    enrolled: bool = False

    def update(self, vector: np.ndarray, t_s: float) -> None:
        """Fold one more observation into the running mean."""
        self.count += 1
        self.mean = self.mean + (vector - self.mean) / self.count
        self.last_seen_s = t_s


# ---------------------------------------------------------------------------
# Outcomes returned to the caller
# ---------------------------------------------------------------------------

@dataclass
class ObservationResult:
    """What the memory did with one observation during the learning phase."""

    profile_id: str
    matched: bool            # folded into an existing profile
    is_new: bool             # created a fresh tentative profile
    distance: float          # distance to the nearest profile (inf if none yet)
    observations: int        # observation count of the profile after the update
    newly_enrolled: bool     # this observation crossed the enrolment threshold
    enrolled: bool           # current enrolled status of the profile


@dataclass
class MatchResult:
    """What the memory concluded about one event during the guarding phase."""

    recognised: bool
    profile_id: Optional[str]      # the enrolled profile that matched, if any
    distance: float                # distance to the closest enrolled profile
    closest_id: Optional[str]      # closest enrolled profile even when no match


# ---------------------------------------------------------------------------
# The pattern memory itself
# ---------------------------------------------------------------------------

class PatternMemory:
    """Online nearest-signature clustering with confirmation and recognition.

    Parameters
    ----------
    feature_scales:
        Per-feature distance scale, one value per feature dimension. Each is the
        expected repeatability of that feature between two sightings of the same
        source. Differences are divided by these before the distance is taken,
        so the threshold is expressed in interpretable "typical measurement
        spreads" rather than raw, incompatible units.
    match_distance:
        Largest normalised distance at which two events are treated as the same
        source. Smaller is stricter.
    enroll_after:
        Number of sightings at which a tentative profile becomes enrolled.
    feature_names:
        Optional labels for the feature dimensions, used only for readable
        summaries.
    """

    def __init__(self, feature_scales: Sequence[float],
                 match_distance: float = 1.5, enroll_after: int = 3,
                 feature_names: Optional[Sequence[str]] = None) -> None:
        self.scales = np.asarray(feature_scales, dtype=float)
        if np.any(self.scales <= 0):
            raise ValueError("feature scales must all be positive")
        self.match_distance = float(match_distance)
        self.enroll_after = int(enroll_after)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.profiles: List[Profile] = []
        self._next_index = 1

    # -- distance ----------------------------------------------------------

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Root-mean-square per-feature difference in units of the feature scale.

        Using the mean across dimensions (rather than the sum) keeps the value
        independent of how many features there are, so a match threshold of,
        say, 1.5 always reads as "about one and a half typical spreads off, on
        average", regardless of the feature count.
        """
        diff = (a - b) / self.scales
        return float(np.sqrt(np.mean(diff * diff)))

    def _nearest(self, vector: np.ndarray,
                 enrolled_only: bool = False) -> tuple[Optional[Profile], float]:
        best: Optional[Profile] = None
        best_dist = float("inf")
        for profile in self.profiles:
            if enrolled_only and not profile.enrolled:
                continue
            dist = self._distance(vector, profile.mean)
            if dist < best_dist:
                best_dist = dist
                best = profile
        return best, best_dist

    # -- internal ----------------------------------------------------------

    def _create_profile(self, vector: np.ndarray, t_s: float) -> Profile:
        profile = Profile(
            profile_id="P-{0:03d}".format(self._next_index),
            mean=np.array(vector, dtype=float),
            count=1, first_seen_s=t_s, last_seen_s=t_s, enrolled=False)
        self._next_index += 1
        self.profiles.append(profile)
        return profile

    # -- learning ----------------------------------------------------------

    def observe(self, vector: np.ndarray, t_s: float) -> ObservationResult:
        """Learn from one event: match it, or remember it as something new.

        If the event is close enough to a profile already in memory (tentative
        or enrolled) it is folded into that profile. Otherwise it founds a new
        tentative profile. Crossing ``enroll_after`` sightings promotes a
        tentative profile to enrolled.
        """
        vector = np.asarray(vector, dtype=float)
        nearest, dist = self._nearest(vector)

        if nearest is not None and dist <= self.match_distance:
            nearest.update(vector, t_s)
            newly_enrolled = False
            if not nearest.enrolled and nearest.count >= self.enroll_after:
                nearest.enrolled = True
                newly_enrolled = True
            return ObservationResult(
                profile_id=nearest.profile_id, matched=True, is_new=False,
                distance=dist, observations=nearest.count,
                newly_enrolled=newly_enrolled, enrolled=nearest.enrolled)

        created = self._create_profile(vector, t_s)
        return ObservationResult(
            profile_id=created.profile_id, matched=False, is_new=True,
            distance=dist, observations=1, newly_enrolled=False, enrolled=False)

    # -- recognition -------------------------------------------------------

    def recognise(self, vector: np.ndarray, t_s: float) -> MatchResult:
        """Decide whether a new event matches a confirmed, enrolled signature.

        This does not mutate memory: the guarding phase only judges. An event
        within ``match_distance`` of an enrolled profile is recognised and can
        be suppressed; anything else is novel and worth an alert. The closest
        enrolled profile and its distance are always reported so the caller can
        show how near a miss it was.
        """
        vector = np.asarray(vector, dtype=float)
        nearest, dist = self._nearest(vector, enrolled_only=True)
        if nearest is None:
            return MatchResult(recognised=False, profile_id=None,
                               distance=float("inf"), closest_id=None)
        if dist <= self.match_distance:
            return MatchResult(recognised=True, profile_id=nearest.profile_id,
                               distance=dist, closest_id=nearest.profile_id)
        return MatchResult(recognised=False, profile_id=None, distance=dist,
                           closest_id=nearest.profile_id)

    # -- maintenance -------------------------------------------------------

    def consolidate(self) -> List[Tuple[str, str]]:
        """Merge enrolled profiles that turned out to describe the same source.

        Occasionally one source founds a second profile, when an early, noisy
        sighting lands just past the match threshold before the profile mean has
        settled. As both profiles accumulate observations their means converge,
        so a pair describing one source ends up within ``match_distance`` of each
        other. Merging such pairs repairs the split. This is safe: two genuinely
        different sources are always farther apart than ``match_distance`` (that
        is why they became separate profiles), so distinct characters are never
        fused.

        Returns the list of (removed_id, kept_id) merges performed, so a caller
        tracking per-profile statistics can fold them together too.
        """
        merges: List[Tuple[str, str]] = []
        changed = True
        while changed:
            changed = False
            enrolled = [p for p in self.profiles if p.enrolled]
            for i in range(len(enrolled)):
                for j in range(i + 1, len(enrolled)):
                    keep, drop = enrolled[i], enrolled[j]
                    if self._distance(keep.mean, drop.mean) <= self.match_distance:
                        total = keep.count + drop.count
                        keep.mean = (keep.mean * keep.count
                                     + drop.mean * drop.count) / total
                        keep.count = total
                        keep.first_seen_s = min(keep.first_seen_s, drop.first_seen_s)
                        keep.last_seen_s = max(keep.last_seen_s, drop.last_seen_s)
                        self.profiles.remove(drop)
                        merges.append((drop.profile_id, keep.profile_id))
                        changed = True
                        break
                if changed:
                    break
        return merges

    def prune(self, now_s: float, tentative_idle_s: float) -> List[str]:
        """Retire tentative profiles not seen for a long time (one-off noise).

        A signature seen once or twice and never again is just noise cluttering
        memory. After ``tentative_idle_s`` with no new sighting, a still-tentative
        profile is dropped. Enrolled profiles are kept (they are confirmed
        regulars). Returns the ids removed.
        """
        removed: List[str] = []
        keep: List[Profile] = []
        for p in self.profiles:
            if not p.enrolled and (now_s - p.last_seen_s) > tentative_idle_s:
                removed.append(p.profile_id)
            else:
                keep.append(p)
        self.profiles = keep
        return removed

    # -- views -------------------------------------------------------------

    def enrolled_profiles(self) -> List[Profile]:
        return [p for p in self.profiles if p.enrolled]

    def tentative_profiles(self) -> List[Profile]:
        return [p for p in self.profiles if not p.enrolled]

    def describe(self, profile: Profile) -> str:
        """Render a profile's mean feature vector as a readable line."""
        if self.feature_names is not None:
            parts = ["{0}={1:.3f}".format(name, value) for name, value
                     in zip(self.feature_names, profile.mean)]
        else:
            parts = ["{0:.3f}".format(value) for value in profile.mean]
        return ", ".join(parts)

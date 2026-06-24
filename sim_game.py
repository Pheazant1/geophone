"""
sim_game.py
===========
Interactive simulation sandbox for the self-learning seismic core.

This is a playable, top-down 2D sandbox built on pygame. You populate a site
with characters (people, animals, vehicles), tune their biometrics, gait and
spontaneity, pick the ground and weather, and fast-forward time. Every time a
character crosses the buried array, the game synthesises the real seismic event,
runs it through the real signal pipeline and the real self-learning pattern
memory from ``pattern_memory.py``, and shows you the database the system builds,
blind, in real time.

The brain in the loop is the genuine one, not a reimplementation, so anything
you learn here about its behaviour, and any tuning you settle on, carries over.
What does not carry over is the learned memory itself, which is correct: the
system is meant to relearn each real site from scratch.

Honesty: ground response is highly site-specific, so the environment numbers in
``environment.py`` are physically reasonable rather than measured truth, and are
tagged grounded or estimate in the panel. The footstep force-to-weight scaling
and gait ranges are literature-grounded; the soil coupling is the part a real
deployment would calibrate.

Run it:

    python sim_game.py
"""

from __future__ import annotations

import datetime
import glob
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from collections import Counter

from data_simulator import (DEFAULT_SAMPLING_RATE_HZ, REF_DISTANCE_M,
                            SENSOR_POSITIONS_M, GeophoneArraySimulator)
from environment import Environment, default_environment, next_preset
from localizer import track_window
from pattern_memory import PatternMemory
from profile_classifier import FEATURE_PAD_S, extract_features
from signal_processor import SignalProcessor
from v1_demo import FEATURE_NAMES, FEATURE_SCALES, _cadence_hz


# ---------------------------------------------------------------------------
# Layout and palette
# ---------------------------------------------------------------------------

WIN_W, WIN_H = 1200, 800
FS = DEFAULT_SAMPLING_RATE_HZ
SITE_M = 30.0

MAP_RECT = pygame.Rect(20, 58, 532, 532)
SCALE = MAP_RECT.width / SITE_M
PANEL_X = 584
BOTTOM_Y = 616

BG = (18, 20, 26)
PANEL = (28, 31, 40)
INK = (228, 232, 240)
DIM = (140, 148, 162)
ACCENT = (90, 180, 255)
GOOD = (110, 210, 140)
WARN = (240, 180, 90)
BAD = (240, 110, 110)
GRID = (44, 48, 60)

CHAR_COLORS = [(95, 200, 255), (250, 160, 90), (150, 220, 120), (235, 130, 235),
               (240, 210, 100), (120, 200, 220), (235, 120, 120), (170, 170, 250)]


def m_to_px(x: float, y: float) -> Tuple[int, int]:
    return (int(MAP_RECT.x + x * SCALE), int(MAP_RECT.y + (SITE_M - y) * SCALE))


# Ground and grid colour per weather, so the site looks like the condition.
ENV_VISUALS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    "dry_firm": ((36, 72, 44), (52, 92, 60)),
    "wet": ((38, 58, 46), (54, 76, 60)),
    "mud": ((74, 56, 38), (96, 76, 52)),
    "snow": ((226, 231, 239), (198, 206, 218)),
    "frozen": ((201, 221, 233), (176, 198, 214)),
    "rain": ((40, 54, 52), (56, 72, 68)),
}


# ---------------------------------------------------------------------------
# Character glyphs (drawn in the character's colour)
# ---------------------------------------------------------------------------

def draw_human(surf, x, y, color, running=False):
    x, y = int(x), int(y)
    spread = 7 if running else 4
    pygame.draw.line(surf, color, (x, y), (x - spread, y + 9), 2)
    pygame.draw.line(surf, color, (x, y), (x + spread, y + 9), 2)
    pygame.draw.line(surf, color, (x, y - 9), (x, y), 2)
    if running:
        pygame.draw.line(surf, color, (x, y - 6), (x + 7, y - 1), 2)
        pygame.draw.line(surf, color, (x, y - 6), (x - 7, y - 9), 2)
    else:
        pygame.draw.line(surf, color, (x, y - 6), (x - 5, y - 1), 2)
        pygame.draw.line(surf, color, (x, y - 6), (x + 5, y - 1), 2)
    pygame.draw.circle(surf, color, (x, y - 13), 4)


def draw_animal(surf, x, y, color):
    x, y = int(x), int(y)
    pygame.draw.ellipse(surf, color, pygame.Rect(x - 10, y - 3, 18, 8))
    for lx in (x - 7, x - 3, x + 2, x + 6):
        pygame.draw.line(surf, color, (lx, y + 4), (lx, y + 10), 2)
    pygame.draw.line(surf, color, (x - 10, y - 1), (x - 15, y - 6), 2)
    pygame.draw.circle(surf, color, (x + 10, y - 3), 4)


def draw_vehicle(surf, x, y, color):
    x, y = int(x), int(y)
    pygame.draw.rect(surf, color, pygame.Rect(x - 14, y - 4, 28, 10), border_radius=3)
    pygame.draw.rect(surf, color, pygame.Rect(x - 7, y - 10, 15, 7), border_radius=2)
    pygame.draw.circle(surf, (22, 24, 30), (x - 8, y + 7), 4)
    pygame.draw.circle(surf, (22, 24, 30), (x + 8, y + 7), 4)
    pygame.draw.circle(surf, color, (x - 8, y + 7), 4, 1)
    pygame.draw.circle(surf, color, (x + 8, y + 7), 4, 1)


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------

@dataclass
class Character:
    name: str
    kind: str                       # "human" | "animal" | "vehicle"
    color: Tuple[int, int, int]
    path_start: Tuple[float, float]
    path_end: Tuple[float, float]
    mass_kg: float = 75.0
    cadence_hz: float = 1.8
    step_freq_hz: float = 30.0
    spontaneity: float = 0.0        # chance of breaking into a run, per crossing
    axle_freq_hz: float = 14.0
    axle_spacing_s: float = 0.45
    interval_s: float = 900.0       # mean time between crossings (visitors)
    resident: bool = False          # lives here: moves often, all day
    next_cross_s: float = 0.0

    def _gap(self, rng: random.Random) -> float:
        if self.resident:
            return rng.uniform(120.0, 360.0)   # room-to-room movement
        return self.interval_s * rng.uniform(0.6, 1.4)

    def schedule_first(self, now_s: float, rng: random.Random) -> None:
        self.next_cross_s = now_s + rng.uniform(2.0, self._gap(rng))

    def reschedule(self, rng: random.Random) -> None:
        self.next_cross_s += self._gap(rng)


@dataclass
class Crossing:
    """A short on-screen animation of one character moving across the site."""

    character: Character
    born_ms: int
    duration_ms: int
    running: bool
    detected: bool
    path_start: Tuple[float, float]
    path_end: Tuple[float, float]


# ---------------------------------------------------------------------------
# Signature extraction (kept consistent with the chosen environment)
# ---------------------------------------------------------------------------

def _amplitude_factor(distance_m: float, absorption_per_m: float) -> float:
    d = max(distance_m, REF_DISTANCE_M)
    return float(np.sqrt(REF_DISTANCE_M / d)
                 * np.exp(-absorption_per_m * (d - REF_DISTANCE_M)))


def extract_signature(result, window, env: Environment) -> Optional[np.ndarray]:
    """Build the label-free feature vector for one detected window.

    Localisation and range correction use the active environment's wave speed
    and absorption so the whole chain is physically consistent. The coupling
    gain is deliberately not corrected out, so weather that muffles or sharpens
    the signal shifts the learned signature, exactly as it would in the field.
    """
    features = extract_features(result, window, FS)
    track = track_window(result.filtered, window.start_idx, window.end_idx, FS,
                         wave_speed_m_s=env.wave_speed_m_s)
    sensor = SENSOR_POSITIONS_M[window.best_channel]
    distance = float(np.hypot(sensor[0] - track.position.x_m,
                              sensor[1] - track.position.y_m))
    reference_amp = features.peak_v / _amplitude_factor(distance, env.absorption_per_m)
    log_ref_amp = float(np.log10(max(reference_amp, 1e-9)))

    pad = int(round(FEATURE_PAD_S * FS))
    s = max(0, window.start_idx - pad)
    e = min(result.envelope.shape[1], window.end_idx + pad)
    cadence = _cadence_hz(result.envelope[window.best_channel][s:e], FS)

    return np.array([log_ref_amp, features.dominant_freq_hz,
                     features.low_freq_ratio, cadence], dtype=float)


# ---------------------------------------------------------------------------
# Simple UI widgets
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label

    def draw(self, surf, font, hot=False):
        col = (60, 70, 92) if not hot else (80, 110, 150)
        pygame.draw.rect(surf, col, self.rect, border_radius=5)
        pygame.draw.rect(surf, (90, 100, 120), self.rect, 1, border_radius=5)
        t = font.render(self.label, True, INK)
        surf.blit(t, (self.rect.centerx - t.get_width() // 2,
                      self.rect.centery - t.get_height() // 2))

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)


class Slider:
    def __init__(self, x, y, w, vmin, vmax, value, label, attr, fmt="{:.0f}"):
        self.rect = pygame.Rect(x, y, w, 16)
        self.vmin, self.vmax = vmin, vmax
        self.value = value
        self.label = label
        self.attr = attr
        self.fmt = fmt
        self.dragging = False

    def _knob_x(self):
        frac = (self.value - self.vmin) / (self.vmax - self.vmin)
        return int(self.rect.x + frac * self.rect.width)

    def draw(self, surf, font):
        cap = "{0}: {1}".format(self.label, self.fmt.format(self.value))
        surf.blit(font.render(cap, True, DIM), (self.rect.x, self.rect.y - 16))
        pygame.draw.line(surf, GRID, (self.rect.x, self.rect.centery),
                         (self.rect.right, self.rect.centery), 3)
        kx = self._knob_x()
        pygame.draw.circle(surf, ACCENT, (kx, self.rect.centery), 7)

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            hit = pygame.Rect(self.rect.x - 8, self.rect.y - 6,
                              self.rect.width + 16, self.rect.height + 12)
            if hit.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if self.dragging and event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            frac = (event.pos[0] - self.rect.x) / max(1, self.rect.width)
            frac = min(1.0, max(0.0, frac))
            self.value = self.vmin + frac * (self.vmax - self.vmin)
            return True
        return False


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------

SPEEDS = [0, 1, 10, 60, 300, 1800]
SPEED_LABELS = ["paused", "1x", "10x", "60x", "300x", "1800x"]

# Hours a resident is awake and moving; outside this they are asleep (no events).
WAKE_START_H = 7
WAKE_END_H = 23
GOLD = (240, 210, 90)

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")


def _fmt_runtime(seconds: int) -> str:
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return "{0}h {1:02d}m".format(h, m) if h else "{0}m {1:02d}s".format(m, s)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Seismic Self-Learning Sandbox")
        self.design = (WIN_W, WIN_H)
        self.fullscreen = False
        self.windowed_size = (1120, 740)
        self._scale = 1.0
        self._off = (0, 0)
        self.frame_dt = 0.0
        self._make_window()
        self.canvas = pygame.Surface(self.design)
        self.screen = self.canvas            # all drawing targets the canvas
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 18, bold=True)

        self.rng = random.Random(1)
        self.rng_vis = random.Random(99)
        self.particles = [[self.rng_vis.uniform(MAP_RECT.left, MAP_RECT.right),
                           self.rng_vis.uniform(MAP_RECT.top, MAP_RECT.bottom),
                           self.rng_vis.uniform(0.4, 1.0)] for _ in range(90)]
        self.np_seed = 1000
        self.sim = GeophoneArraySimulator(sampling_rate_hz=FS, seed=7)
        self.processor = SignalProcessor(fs=FS, merge_gap_s=1.2)
        self.env = default_environment()

        self.characters: List[Character] = []
        self.selected: Optional[Character] = None
        self.sliders: List[Slider] = []
        self.crossings: List[Crossing] = []
        self.log: List[Tuple[str, Tuple[int, int, int]]] = []

        self.clock_s = 0.0
        self.speed_idx = 3
        self.char_counter = 0
        self._last_running = False
        self.start_ticks = pygame.time.get_ticks()

        self.mode = "sim"               # sim | report | history | viewer
        self.flagged: List[dict] = []   # novel signatures seen after warm-up
        self.history: List[Tuple[str, dict]] = []
        self.viewer_report: Optional[dict] = None
        self._history_rects: List[Tuple[pygame.Rect, dict]] = []
        self._saved_on_exit = False

        self._build_memory()
        self._build_buttons()
        self._seed_default_cast()

    # -- window and scaling ------------------------------------------------

    def _make_window(self):
        if self.fullscreen:
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.window = pygame.display.set_mode(self.windowed_size,
                                                  pygame.RESIZABLE)

    def _present(self):
        """Scale the fixed-layout canvas to fill the window, keeping aspect."""
        ww, wh = self.window.get_size()
        dw, dh = self.design
        self._scale = min(ww / dw, wh / dh)
        sw, sh = int(dw * self._scale), int(dh * self._scale)
        self._off = ((ww - sw) // 2, (wh - sh) // 2)
        self.window.fill((0, 0, 0))
        self.window.blit(pygame.transform.smoothscale(self.canvas, (sw, sh)),
                         self._off)
        pygame.display.flip()

    def _mouse_canvas(self):
        mx, my = pygame.mouse.get_pos()
        return ((mx - self._off[0]) / self._scale, (my - self._off[1]) / self._scale)

    # -- setup -------------------------------------------------------------

    def _build_memory(self):
        self.memory = PatternMemory(FEATURE_SCALES, match_distance=1.2,
                                    enroll_after=3, feature_names=FEATURE_NAMES)
        self.profile_truth: Dict[str, Counter] = {}

    def _build_buttons(self):
        x = PANEL_X
        self.btn_weather = Button(x, 112, 250, 30, "Weather: cycle >")
        self.btn_reset = Button(x + 470, 112, 110, 30, "Reset AI")
        self.speed_buttons = []
        for i, lab in enumerate(SPEED_LABELS):
            self.speed_buttons.append(Button(x + i * 84, 204, 78, 26, lab))
        self.btn_human = Button(x, 250, 130, 28, "+ Human")
        self.btn_animal = Button(x + 138, 250, 130, 28, "+ Animal")
        self.btn_vehicle = Button(x + 276, 250, 130, 28, "+ Vehicle")
        self.btn_remove = Button(x + 470, 250, 110, 28, "Remove")
        self.btn_resident = Button(x, 544, 250, 28, "Resident: off")

        # session controls (right column, lower area)
        sx = x + 300
        self.btn_save = Button(sx, 470, 150, 28, "Save session")
        self.btn_end = Button(sx, 504, 200, 28, "End + report")
        self.btn_history = Button(sx, 538, 200, 28, "Past sessions")

        # report / history screen buttons
        by = WIN_H - 52
        self.btn_r_save = Button(40, by, 150, 32, "Save report")
        self.btn_r_resume = Button(200, by, 150, 32, "Resume sim")
        self.btn_r_history = Button(360, by, 180, 32, "Past sessions")
        self.btn_r_quit = Button(550, by, 120, 32, "Quit")
        self.btn_back = Button(40, by, 150, 32, "Back")

    def _seed_default_cast(self):
        self.add_character("human", mass=80, cadence=1.6, step_freq=30,
                           start=(8, 10), end=(14, 16))
        self.add_character("human", mass=78, cadence=2.2, step_freq=32,
                           start=(22, 9), end=(17, 15))
        self.add_character("animal", mass=45, cadence=3.0, step_freq=40,
                           start=(11, 11), end=(18, 15))

    # -- character management ----------------------------------------------

    def add_character(self, kind, mass=None, cadence=None, step_freq=None,
                      start=None, end=None):
        self.char_counter += 1
        color = CHAR_COLORS[(self.char_counter - 1) % len(CHAR_COLORS)]
        if start is None:
            start = (self.rng.uniform(2, 12), self.rng.uniform(2, 28))
            end = (self.rng.uniform(18, 28), self.rng.uniform(2, 28))
        defaults = {
            "human": dict(mass=75, cadence=1.8, step_freq=30, name="Person"),
            "animal": dict(mass=40, cadence=3.0, step_freq=40, name="Animal"),
            "vehicle": dict(mass=1500, cadence=1.8, step_freq=30, name="Vehicle"),
        }[kind]
        ch = Character(
            name="{0}-{1}".format(defaults["name"], self.char_counter),
            kind=kind, color=color, path_start=start, path_end=end,
            mass_kg=mass if mass is not None else defaults["mass"],
            cadence_hz=cadence if cadence is not None else defaults["cadence"],
            step_freq_hz=step_freq if step_freq is not None else defaults["step_freq"],
            interval_s=self.rng.uniform(600, 1200))
        ch.schedule_first(self.clock_s, self.rng)
        self.characters.append(ch)
        self.select(ch)

    def select(self, ch: Optional[Character]):
        self.selected = ch
        self.sliders = []
        if ch is None:
            return
        x, y, w = PANEL_X, 340, 250
        if ch.kind in ("human", "animal"):
            mmax = 150 if ch.kind == "human" else 90
            self.sliders = [
                Slider(x, y, w, 5, mmax, ch.mass_kg, "Mass (kg)", "mass_kg"),
                Slider(x, y + 50, w, 1.0, 4.0, ch.cadence_hz, "Cadence (steps/s)",
                       "cadence_hz", "{:.2f}"),
                Slider(x, y + 100, w, 20, 45, ch.step_freq_hz, "Foot freq (Hz)",
                       "step_freq_hz"),
                Slider(x, y + 150, w, 0.0, 1.0, ch.spontaneity, "Run chance",
                       "spontaneity", "{:.0%}"),
            ]
        else:
            self.sliders = [
                Slider(x, y, w, 400, 3000, ch.mass_kg, "Weight (kg)", "mass_kg"),
                Slider(x, y + 50, w, 8, 20, ch.axle_freq_hz, "Axle freq (Hz)",
                       "axle_freq_hz"),
                Slider(x, y + 100, w, 0.3, 0.8, ch.axle_spacing_s, "Axle gap (s)",
                       "axle_spacing_s", "{:.2f}"),
            ]

    # -- roaming paths across the whole site -------------------------------

    def _edge_point(self) -> Tuple[float, float]:
        """A random point just outside one edge of the site, for a pass-through."""
        lo, hi = -3.0, SITE_M + 3.0
        side = self.rng.randint(0, 3)
        if side == 0:
            return (self.rng.uniform(lo, hi), lo)
        if side == 1:
            return (self.rng.uniform(lo, hi), hi)
        if side == 2:
            return (lo, self.rng.uniform(lo, hi))
        return (hi, self.rng.uniform(lo, hi))

    def _roam_path(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """A varied path that uses the whole area: edge to edge, or to the centre.

        Most crossings enter from one side and leave by another, sweeping across
        the array. About a third head into the interior instead, modelling
        someone moving to the middle of the site rather than straight through.
        """
        start = self._edge_point()
        if self.rng.random() < 0.35:
            end = (self.rng.uniform(6, SITE_M - 6), self.rng.uniform(6, SITE_M - 6))
        else:
            end = self._edge_point()
        return start, end

    # -- the seismic event for one crossing --------------------------------

    def _synthesize_and_learn(self, ch: Character):
        running = (ch.kind in ("human", "animal")
                   and self.rng.random() < ch.spontaneity)
        self._last_running = running
        self.np_seed += 1
        sim = GeophoneArraySimulator(
            sampling_rate_hz=FS, seed=self.np_seed,
            wave_speed_m_s=self.env.wave_speed_m_s,
            absorption_per_m=self.env.absorption_per_m)
        feed = sim.baseline(duration_s=12.0, noise_floor_v=self.env.noise_floor_v)
        start_xy, end_xy = self._roam_path()
        ps = np.asarray(start_xy, dtype=float)
        pe = np.asarray(end_xy, dtype=float)

        gain = self.env.coupling_gain * (2.0 if running else 1.0)
        if ch.kind in ("human", "animal"):
            sim.add_footstep_train(
                feed, start_s=3.0, path_start_xy=ps, path_end_xy=pe,
                num_steps=8 if ch.kind == "animal" else 6,
                cadence_hz=float(ch.cadence_hz * (1.6 if running else 1.0)),
                mass_kg=float(ch.mass_kg * gain),
                freq_hz=float(ch.step_freq_hz * (1.1 if running else 1.0)),
                label=ch.name)
        else:
            half = ch.mass_kg * self.env.coupling_gain
            sim.add_vehicle_pass(
                feed, start_s=3.0, path_start_xy=ps, path_end_xy=pe,
                axle_masses=(half * 0.45, half * 0.55),
                axle_spacing_s=ch.axle_spacing_s, axle_freq_hz=ch.axle_freq_hz,
                label=ch.name)

        result = self.processor.process(feed)
        if not result.windows:
            self._log("{0}  {1} -> too faint to detect".format(
                self._timestr(), ch.name), WARN)
            return False, ps, pe

        window = max(result.windows, key=lambda w: w.peak_v)
        vector = extract_signature(result, window, self.env)
        obs = self.memory.observe(vector, self.clock_s)
        truth_name = ch.name + (" (running)" if running else "")
        self.profile_truth.setdefault(obs.profile_id, Counter())[ch.name] += 1

        # Consolidate any same-source split and fold the tallies together.
        for removed, kept in self.memory.consolidate():
            if removed in self.profile_truth:
                self.profile_truth.setdefault(kept, Counter()).update(
                    self.profile_truth.pop(removed))

        run_tag = " [RUN]" if running else ""
        # Flag only signatures that are clearly unlike anything already known,
        # once the regulars are established, so a regular's momentarily odd
        # reading is not mistaken for an intruder.
        warm = (len(self.memory.enrolled_profiles()) >= 2 and self.clock_s > 3600
                and obs.distance > 1.6)
        if obs.is_new and warm:
            self.flagged.append({"time": self._timestr(), "identity": ch.name,
                                 "distance": round(obs.distance, 2)})
            self._log("{0}  {1}{2} -> NEW {3}  *** flagged: novel ***".format(
                self._timestr(), ch.name, run_tag, obs.profile_id), BAD)
        elif obs.is_new:
            self._log("{0}  {1}{2} -> NEW {3}".format(
                self._timestr(), ch.name, run_tag, obs.profile_id), ACCENT)
        elif obs.newly_enrolled:
            self._log("{0}  {1}{2} -> {3} CONFIRMED".format(
                self._timestr(), ch.name, run_tag, obs.profile_id), GOOD)
        else:
            self._log("{0}  {1}{2} -> {3} (d={4:.2f})".format(
                self._timestr(), ch.name, run_tag, obs.profile_id, obs.distance), INK)
        return True, ps, pe

    # -- update loop -------------------------------------------------------

    def update(self, dt_real: float):
        speed = SPEEDS[self.speed_idx]
        self.clock_s += dt_real * speed
        if speed == 0:
            return
        for ch in self.characters:
            fired = 0
            while self.clock_s >= ch.next_cross_s and fired < 3:
                if ch.resident and not self._is_awake():
                    ch.next_cross_s = self._next_wake()    # residents sleep at night
                    break
                detected, ps, pe = self._synthesize_and_learn(ch)
                running = self._last_running
                self.crossings.append(Crossing(
                    ch, pygame.time.get_ticks(), 1100, running, detected,
                    (float(ps[0]), float(ps[1])), (float(pe[0]), float(pe[1]))))
                ch.reschedule(self.rng)
                fired += 1
        now = pygame.time.get_ticks()
        self.crossings = [c for c in self.crossings
                          if now - c.born_ms < c.duration_ms]

    # -- helpers -----------------------------------------------------------

    def _timestr(self) -> str:
        day = int(self.clock_s // 86400) + 1
        tod = int(self.clock_s % 86400)
        return "D{0:02d} {1:02d}:{2:02d}".format(day, tod // 3600, (tod % 3600) // 60)

    def _is_awake(self) -> bool:
        hour = (self.clock_s % 86400) / 3600.0
        return WAKE_START_H <= hour < WAKE_END_H

    def _next_wake(self) -> float:
        day = int(self.clock_s // 86400)
        today = day * 86400 + WAKE_START_H * 3600
        return today if self.clock_s < today else (day + 1) * 86400 + WAKE_START_H * 3600

    def _log(self, text, color=INK):
        self.log.append((text, color))
        self.log = self.log[-9:]

    # -- sessions and reporting --------------------------------------------

    def _identity(self, pid):
        c = self.profile_truth.get(pid, Counter())
        if not c:
            return ("unknown", 0.0)
        dom, n = c.most_common(1)[0]
        return (dom, n / sum(c.values()))

    def build_report(self) -> dict:
        enrolled = self.memory.enrolled_profiles()
        tentative = self.memory.tentative_profiles()
        regulars = []
        for p in enrolled:
            ident, pur = self._identity(p.profile_id)
            res = any(ch.name == ident and ch.resident for ch in self.characters)
            regulars.append({"id": p.profile_id, "obs": p.count,
                             "identity": ident, "resident": res,
                             "purity": round(pur, 2)})
        regulars.sort(key=lambda r: -r["obs"])
        suspicious = [{"id": p.profile_id, "obs": p.count,
                       "identity": self._identity(p.profile_id)[0]}
                      for p in tentative]
        return {
            "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sim_time": self._timestr(),
            "sim_days": round(self.clock_s / 86400, 2),
            "real_run_s": int((pygame.time.get_ticks() - self.start_ticks) / 1000),
            "weather": self.env.name,
            "crossings": sum(p.count for p in self.memory.profiles),
            "regulars": regulars,
            "suspicious_profiles": suspicious,
            "flagged_events": self.flagged[-40:],
            "cast": [{"name": c.name, "kind": c.kind, "resident": c.resident,
                      "mass_kg": round(c.mass_kg, 1),
                      "cadence_hz": round(c.cadence_hz, 2)} for c in self.characters],
        }

    def save_session(self) -> str:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        name = "session_{0}.json".format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        with open(os.path.join(SESSIONS_DIR, name), "w", encoding="utf-8") as f:
            json.dump(self.build_report(), f, indent=2)
        self._log("saved to sessions/" + name, GOOD)
        return name

    def _autosave(self):
        if not self._saved_on_exit and any(p.count for p in self.memory.profiles):
            self.save_session()
            self._saved_on_exit = True

    def _enter_history(self):
        self.history = []
        if os.path.isdir(SESSIONS_DIR):
            for path in sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.json")),
                               reverse=True):
                try:
                    with open(path, encoding="utf-8") as f:
                        self.history.append((path, json.load(f)))
                except (OSError, ValueError):
                    continue
        self.mode = "history"

    # -- drawing -----------------------------------------------------------

    def draw(self):
        if self.mode == "report":
            self._draw_report(self.build_report(), live=True)
            return
        if self.mode == "history":
            self._draw_history()
            return
        if self.mode == "viewer":
            self._draw_report(self.viewer_report, live=False)
            return
        self.screen.fill(BG)
        self._draw_header()
        self._draw_map()
        self._draw_controls()
        self._draw_database()
        self._draw_log()

    def _draw_report(self, rep, live):
        self.screen.fill(BG)
        mp = self._mouse_canvas()
        head = ("SESSION REPORT" if live
                else "PAST SESSION   saved {0}".format(rep.get("saved_at", "")))
        self.screen.blit(self.big.render(head, True, INK), (40, 24))
        self.screen.blit(self.font.render(
            "weather {0}   |   sim {1} ({2} days)   |   real run {3}   |   "
            "crossings learned {4}".format(
                rep["weather"], rep["sim_time"], rep["sim_days"],
                _fmt_runtime(rep["real_run_s"]), rep["crossings"]),
            True, DIM), (40, 56))

        # left: the learned database (regulars)
        self.screen.blit(self.big.render(
            "Known regulars the system learned", True, GOOD), (40, 96))
        y = 128
        if not rep["regulars"]:
            self.screen.blit(self.font.render("(none enrolled yet)", True, DIM),
                             (50, y))
        for r in rep["regulars"][:22]:
            tag = "  [RESIDENT]" if r.get("resident") else ""
            col = GOLD if r.get("resident") else GOOD
            self.screen.blit(self.font.render(
                "{0}  {1:>5} obs   {2}{3}   ({4:.0%} pure)".format(
                    r["id"], r["obs"], r["identity"], tag, r["purity"]),
                True, col), (50, y))
            y += 22

        # right: flagged / suspicious
        rx = 624
        self.screen.blit(self.big.render(
            "Flagged / one-off activity", True, WARN), (rx, 96))
        self.screen.blit(self.small.render(
            "novel signatures after the regulars were known, and one-offs that "
            "never recurred", True, DIM), (rx, 122))
        y = 144
        if not rep["flagged_events"]:
            self.screen.blit(self.font.render("none flagged", True, DIM), (rx + 10, y))
            y += 22
        for ev in rep["flagged_events"][-12:]:
            self.screen.blit(self.font.render(
                "{0}  novel signature (closest d={1})  [was {2}]".format(
                    ev["time"], ev["distance"], ev["identity"]), True, BAD),
                (rx + 10, y))
            y += 20
        y += 14
        self.screen.blit(self.font.render(
            "Unconfirmed one-off signatures:", True, DIM), (rx, y))
        y += 22
        if not rep["suspicious_profiles"]:
            self.screen.blit(self.small.render("none", True, DIM), (rx + 10, y))
        for s in rep["suspicious_profiles"][:10]:
            self.screen.blit(self.small.render(
                "{0}  {1} obs  (was {2})".format(s["id"], s["obs"], s["identity"]),
                True, WARN), (rx + 10, y))
            y += 18

        if live:
            for b in (self.btn_r_save, self.btn_r_resume, self.btn_r_history,
                      self.btn_r_quit):
                b.draw(self.screen, self.font, b.hit(mp))
        else:
            self.btn_back.draw(self.screen, self.font, self.btn_back.hit(mp))

    def _draw_history(self):
        self.screen.fill(BG)
        mp = self._mouse_canvas()
        self.screen.blit(self.big.render("PAST SESSIONS (click one to view)",
                                         True, INK), (40, 24))
        self._history_rects = []
        if not self.history:
            self.screen.blit(self.font.render(
                "no saved sessions yet. Use 'Save session' or 'End + report'.",
                True, DIM), (40, 80))
        y = 76
        for path, rep in self.history[:24]:
            rect = pygame.Rect(40, y, WIN_W - 80, 28)
            hot = rect.collidepoint(mp)
            pygame.draw.rect(self.screen, (60, 66, 84) if hot else PANEL, rect,
                             border_radius=4)
            self.screen.blit(self.font.render(
                "{0}   |   {1}   |   {2} regulars, {3} flagged   |   "
                "{4} days, {5} crossings".format(
                    rep.get("saved_at", "?"), rep.get("weather", "?"),
                    len(rep.get("regulars", [])), len(rep.get("flagged_events", [])),
                    rep.get("sim_days", "?"), rep.get("crossings", "?")),
                True, INK), (52, y + 5))
            self._history_rects.append((rect, rep))
            y += 32
        self.btn_back.draw(self.screen, self.font, self.btn_back.hit(mp))

    def _draw_header(self):
        learned = sum(p.count for p in self.memory.profiles)
        real = (pygame.time.get_ticks() - self.start_ticks) // 1000
        rh, rm, rs = real // 3600, (real % 3600) // 60, real % 60
        realstr = ("{0}h {1:02d}m".format(rh, rm) if rh
                   else "{0}m {1:02d}s".format(rm, rs))
        night = "" if self._is_awake() else "  (night)"
        self.screen.blit(self.big.render(
            "Seismic Self-Learning Sandbox", True, INK), (20, 16))
        self.screen.blit(self.font.render(
            "sim {0}{1}    speed {2}    learned {3}    real run {4}".format(
                self._timestr(), night, SPEED_LABELS[self.speed_idx], learned,
                realstr), True, DIM), (360, 22))
        self.screen.blit(self.small.render(
            "F=fullscreen  ESC=quit", True, DIM), (WIN_W - 190, 24))

    def _draw_glyph(self, ch, p, running):
        if ch.kind == "vehicle":
            draw_vehicle(self.screen, p[0], p[1], ch.color)
        elif ch.kind == "animal":
            draw_animal(self.screen, p[0], p[1], ch.color)
        else:
            draw_human(self.screen, p[0], p[1], ch.color, running)

    def _draw_weather(self):
        mode = self.env.key
        if mode not in ("rain", "snow"):
            return
        for pt in self.particles:
            if mode == "rain":
                pt[1] += 620 * pt[2] * self.frame_dt
                pygame.draw.line(self.screen, (140, 165, 210),
                                 (pt[0], pt[1]), (pt[0] - 2, pt[1] + 9), 1)
            else:
                pt[1] += 110 * pt[2] * self.frame_dt
                pt[0] += math.sin(pt[1] * 0.04) * 0.4
                pygame.draw.circle(self.screen, (240, 243, 250),
                                   (int(pt[0]), int(pt[1])), 2)
            if pt[1] > MAP_RECT.bottom:
                pt[1] = MAP_RECT.top
                pt[0] = self.rng_vis.uniform(MAP_RECT.left, MAP_RECT.right)

    def _draw_map(self):
        ground, grid = ENV_VISUALS.get(self.env.key, ((36, 72, 44), (52, 92, 60)))
        light_ground = self.env.key in ("snow", "frozen")
        pygame.draw.rect(self.screen, ground, MAP_RECT, border_radius=6)
        prev = self.screen.get_clip()
        self.screen.set_clip(MAP_RECT)
        for g in range(0, 31, 5):
            pygame.draw.line(self.screen, grid, m_to_px(g, 0), m_to_px(g, 30), 1)
            pygame.draw.line(self.screen, grid, m_to_px(0, g), m_to_px(30, g), 1)
        self._draw_weather()
        # active crossings: each draws its own path this time, plus a figure
        now = pygame.time.get_ticks()
        for c in self.crossings:
            frac = min(1.0, (now - c.born_ms) / c.duration_ms)
            faint = tuple(int(v * (0.55 if light_ground else 0.4)) for v in c.character.color)
            pygame.draw.line(self.screen, faint, m_to_px(*c.path_start),
                             m_to_px(*c.path_end), 1)
            x = c.path_start[0] + (c.path_end[0] - c.path_start[0]) * frac
            y = c.path_start[1] + (c.path_end[1] - c.path_start[1]) * frac
            p = m_to_px(x, y)
            if not c.detected:
                pygame.draw.circle(self.screen, BAD, p, 16, 2)
            self._draw_glyph(c.character, p, c.running)
            if c.character.resident:
                pygame.draw.circle(self.screen, GOLD, (p[0], p[1] - 22), 3)
        self.screen.set_clip(prev)
        # sensors on top
        for i, (sx, sy) in enumerate(SENSOR_POSITIONS_M):
            p = m_to_px(sx, sy)
            pygame.draw.rect(self.screen, ACCENT,
                             pygame.Rect(p[0] - 6, p[1] - 6, 12, 12), border_radius=2)
            lx = p[0] + 8 if sx < SITE_M else p[0] - 24
            ly = p[1] - 18 if sy < SITE_M else p[1] + 6
            label_col = (40, 44, 54) if light_ground else DIM
            self.screen.blit(self.small.render("S{0}".format(i), True, label_col),
                             (lx, ly))
        self.screen.blit(self.small.render(
            "30 m square array  |  S = geophone  |  figure = live crossing "
            "(red ring = missed)", True, DIM), (MAP_RECT.x, MAP_RECT.bottom + 8))

    def _draw_controls(self):
        x = PANEL_X
        mp = self._mouse_canvas()
        self.screen.blit(self.big.render("Controls", True, INK), (x, 58))

        # ground and weather
        self.screen.blit(self.font.render("Ground and weather", True, DIM), (x, 92))
        self.btn_weather.label = "Weather: {0} >".format(self.env.name)
        self.btn_weather.draw(self.screen, self.font, self.btn_weather.hit(mp))
        self.btn_reset.draw(self.screen, self.font, self.btn_reset.hit(mp))
        est = sum(1 for _, (lvl, _) in self.env.confidence.items() if lvl == "estimate")
        self.screen.blit(self.small.render(
            "{0}  ({1})".format(self.env.name, self.env.summary), True, DIM),
            (x, 150))
        self.screen.blit(self.small.render(
            "[{0} of {1} values are estimates]".format(est, len(self.env.confidence)),
            True, WARN), (x, 166))

        # speed
        self.screen.blit(self.font.render("Speed (space=pause, +/-)", True, DIM),
                         (x, 184))
        for i, b in enumerate(self.speed_buttons):
            b.draw(self.screen, self.small, i == self.speed_idx)

        # add / remove
        for b in (self.btn_human, self.btn_animal, self.btn_vehicle, self.btn_remove):
            b.draw(self.screen, self.font, b.hit(mp))

        pygame.draw.line(self.screen, GRID, (x, 292), (WIN_W - 20, 292), 1)

        # editor (left)
        if self.selected is not None:
            self.screen.blit(self.font.render(
                "Editing: {0}  ({1})".format(self.selected.name, self.selected.kind),
                True, self.selected.color), (x, 304))
            for s in self.sliders:
                s.draw(self.screen, self.font)
            self.btn_resident.label = "Resident: {0}".format(
                "ON (lives here)" if self.selected.resident else "off (visitor)")
            self.btn_resident.draw(self.screen, self.font,
                                   self.selected.resident or self.btn_resident.hit(mp))
        else:
            self.screen.blit(self.font.render(
                "Add or pick a character to edit ->", True, DIM), (x, 304))

        # roster (right)
        rx = x + 300
        self.screen.blit(self.font.render("Roster (click to edit)", True, DIM),
                         (rx, 304))
        for i, ch in enumerate(self.characters):
            r = pygame.Rect(rx, 330 + i * 24, 280, 22)
            sel = ch is self.selected
            pygame.draw.rect(self.screen, (60, 66, 84) if sel else PANEL, r,
                             border_radius=4)
            pygame.draw.circle(self.screen, ch.color, (rx + 12, r.centery), 5)
            self.screen.blit(self.small.render(
                "{0}  {1}".format(ch.name, ch.kind), True, INK),
                (rx + 26, r.y + 4))
            if ch.resident:
                self.screen.blit(self.small.render("resident", True, GOLD),
                                 (r.right - 64, r.y + 4))

        # session controls
        self.screen.blit(self.font.render("Session", True, DIM), (x + 300, 446))
        for b in (self.btn_save, self.btn_end, self.btn_history):
            b.draw(self.screen, self.font, b.hit(mp))

    def _draw_database(self):
        x = 20
        pygame.draw.rect(self.screen, PANEL,
                         pygame.Rect(x, BOTTOM_Y, 700, WIN_H - BOTTOM_Y - 12),
                         border_radius=6)
        self.screen.blit(self.big.render(
            "AI database (built blind)", True, INK), (x + 10, BOTTOM_Y + 8))
        enrolled = self.memory.enrolled_profiles()
        tentative = self.memory.tentative_profiles()
        self.screen.blit(self.font.render(
            "enrolled: {0}    tentative/noise: {1}".format(
                len(enrolled), len(tentative)), True, DIM),
            (x + 320, BOTTOM_Y + 12))

        yy = BOTTOM_Y + 36
        for p in enrolled[:5]:
            counts = self.profile_truth.get(p.profile_id, Counter())
            total = sum(counts.values())
            if total:
                dom, dn = counts.most_common(1)[0]
                purity = dn / total
            else:
                dom, purity = "?", 0.0
            col = GOOD if purity >= 0.9 else WARN
            self.screen.blit(self.font.render(
                "{0}  {1:>3} obs  ->  {2}  ({3:.0%} pure)".format(
                    p.profile_id, p.count, dom, purity), True, col), (x + 14, yy))
            yy += 20
        if tentative:
            self.screen.blit(self.small.render(
                "tentative: " + ", ".join(t.profile_id for t in tentative[:8]),
                True, DIM), (x + 14, yy + 2))

    def _draw_log(self):
        x = 740
        pygame.draw.rect(self.screen, PANEL,
                         pygame.Rect(x, BOTTOM_Y, WIN_W - x - 20, WIN_H - BOTTOM_Y - 12),
                         border_radius=6)
        self.screen.blit(self.big.render("Live events", True, INK),
                         (x + 10, BOTTOM_Y + 8))
        yy = BOTTOM_Y + 34
        for text, color in self.log:
            self.screen.blit(self.small.render(text, True, color), (x + 12, yy))
            yy += 16

    # -- input -------------------------------------------------------------

    def handle(self, event):
        if event.type == pygame.QUIT:
            self._autosave()
            return False
        if event.type == pygame.VIDEORESIZE and not self.fullscreen:
            self.windowed_size = (max(640, event.w), max(480, event.h))
            self.window = pygame.display.set_mode(self.windowed_size,
                                                  pygame.RESIZABLE)
            return True
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN,
                          pygame.MOUSEBUTTONUP):
            cx = (event.pos[0] - self._off[0]) / self._scale
            cy = (event.pos[1] - self._off[1]) / self._scale
            event = pygame.event.Event(event.type,
                                       {**event.dict, "pos": (int(cx), int(cy))})
        if self.mode == "report":
            return self._handle_report(event)
        if self.mode == "history":
            return self._handle_history(event)
        if self.mode == "viewer":
            return self._handle_viewer(event)
        return self._handle_sim(event)

    def _handle_sim(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.speed_idx = 0 if self.speed_idx else 3
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                self.speed_idx = min(len(SPEEDS) - 1, self.speed_idx + 1)
            elif event.key == pygame.K_MINUS:
                self.speed_idx = max(0, self.speed_idx - 1)
            elif event.key == pygame.K_f:
                self.fullscreen = not self.fullscreen
                self._make_window()
            elif event.key == pygame.K_ESCAPE:
                self.mode = "report"      # ending the game shows the summary
        for s in self.sliders:
            if s.handle(event) and self.selected is not None:
                setattr(self.selected, s.attr, s.value)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._click(event.pos)
        return True

    def _handle_report(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._autosave()
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            p = event.pos
            if self.btn_r_save.hit(p):
                self.save_session()
            elif self.btn_r_resume.hit(p):
                self.mode = "sim"
            elif self.btn_r_history.hit(p):
                self._enter_history()
            elif self.btn_r_quit.hit(p):
                self._autosave()
                return False
        return True

    def _handle_history(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.mode = "sim"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.btn_back.hit(event.pos):
                self.mode = "sim"
                return True
            for rect, rep in self._history_rects:
                if rect.collidepoint(event.pos):
                    self.viewer_report = rep
                    self.mode = "viewer"
                    return True
        return True

    def _handle_viewer(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.mode = "history"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.btn_back.hit(event.pos):
                self.mode = "history"
        return True

    def _click(self, pos):
        if self.btn_save.hit(pos):
            self.save_session()
            return
        if self.btn_end.hit(pos):
            self.mode = "report"
            return
        if self.btn_history.hit(pos):
            self._enter_history()
            return
        if self.btn_weather.hit(pos):
            self.env = next_preset(self.env.key)
            return
        if self.btn_reset.hit(pos):
            self._build_memory()
            self._log("AI memory reset", WARN)
            return
        for i, b in enumerate(self.speed_buttons):
            if b.hit(pos):
                self.speed_idx = i
                return
        if self.btn_human.hit(pos):
            self.add_character("human"); return
        if self.btn_animal.hit(pos):
            self.add_character("animal"); return
        if self.btn_vehicle.hit(pos):
            self.add_character("vehicle"); return
        if self.btn_remove.hit(pos) and self.selected is not None:
            self.characters.remove(self.selected)
            self.select(self.characters[0] if self.characters else None)
            return
        if self.selected is not None and self.btn_resident.hit(pos):
            self.selected.resident = not self.selected.resident
            self.selected.next_cross_s = self.clock_s + 1.0
            return
        rx = PANEL_X + 300
        for i, ch in enumerate(self.characters):
            r = pygame.Rect(rx, 330 + i * 24, 280, 22)
            if r.collidepoint(pos):
                self.select(ch)
                return

    # -- main loop ---------------------------------------------------------

    def run(self):
        running = True
        while running:
            self.frame_dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if not self.handle(event):
                    running = False
            self.update(self.frame_dt)
            self.draw()
            self._present()
        pygame.quit()


def main():
    Game().run()


if __name__ == "__main__":
    main()

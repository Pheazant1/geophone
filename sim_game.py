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

import csv
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


# A character is either an occasional "visitor" (appears at any hour, rarely) or
# a "mover" who follows a custom daily schedule: a 24-slot on/off list of the
# hours they are active. Gaps in the schedule are natural breaks (a lunch run, a
# shift, time off site). Movers cross often during active hours.
VISITOR_GAP = (600.0, 1200.0)
MOVER_GAP = (120.0, 360.0)


def make_schedule(*ranges) -> List[bool]:
    """Build a 24-hour on/off list from (start_hour, end_hour) ranges.

    A start later than the end wraps past midnight, e.g. (22, 6) is overnight.
    """
    sched = [False] * 24
    for a, b in ranges:
        for h in range(24):
            on = (a <= h < b) if a <= b else (h >= a or h < b)
            if on:
                sched[h] = True
    return sched


SCHEDULE_PRESETS = {
    "home": make_schedule((7, 23)),
    "office": make_schedule((9, 17)),
    "night": make_schedule((22, 6)),
    "all": [True] * 24,
}


# Surfaces a footstep can land on. Harder surfaces couple more energy and ring
# higher (less damping); soft ground absorbs. freq_mult and amp_mult are honest
# estimates of the direction, the kind a real site would calibrate by walking it.
SURFACE_TYPES = {
    "soil": {"label": "Soil (default)", "freq": 1.0, "amp": 1.0, "color": (40, 72, 44)},
    "pavement": {"label": "Pavement", "freq": 1.25, "amp": 1.4, "color": (118, 120, 128)},
    "concrete": {"label": "Concrete slab", "freq": 1.45, "amp": 1.8, "color": (150, 152, 160)},
    "gravel": {"label": "Gravel", "freq": 1.12, "amp": 0.8, "color": (120, 108, 88)},
    "building": {"label": "Building floor", "freq": 1.55, "amp": 1.9, "color": (96, 84, 72)},
}
SURFACE_BRUSHES = ["pavement", "concrete", "gravel", "building"]


@dataclass
class Surface:
    kind: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


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
    mover: bool = False             # follows a daily schedule, vs occasional visitor
    schedule: List[bool] = field(default_factory=lambda: make_schedule((7, 23)))
    away: bool = False              # temporarily absent (on holiday): no crossings
    is_intruder: bool = False       # ground truth: counts as an intrusion you set up
    active: bool = False            # released into the scene (produces crossings)
    numbered: bool = False          # named, after it first left an impact
    next_cross_s: float = 0.0

    def _gap(self, rng: random.Random) -> float:
        return rng.uniform(*(MOVER_GAP if self.mover else VISITOR_GAP))

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


def extract_signature(result, window, env: Environment,
                      surface_fn=None) -> Optional[np.ndarray]:
    """Build the label-free feature vector for one detected window.

    Localisation and range correction use the active environment's wave speed
    and absorption so the whole chain is physically consistent. The coupling
    gain is deliberately not corrected out, so weather that muffles or sharpens
    the signal shifts the learned signature, exactly as it would in the field.

    If ``surface_fn`` is given (the AI is surface-aware), the surface under the
    localised source is used to undo its frequency and coupling effect, the same
    way range is corrected, so a person reads the same across surfaces. When it
    is None the AI is surface-blind and surfaces shift the signature.
    """
    features = extract_features(result, window, FS)
    track = track_window(result.filtered, window.start_idx, window.end_idx, FS,
                         wave_speed_m_s=env.wave_speed_m_s)
    sensor = SENSOR_POSITIONS_M[window.best_channel]
    distance = float(np.hypot(sensor[0] - track.position.x_m,
                              sensor[1] - track.position.y_m))
    reference_amp = features.peak_v / _amplitude_factor(distance, env.absorption_per_m)
    dominant_freq = features.dominant_freq_hz

    if surface_fn is not None:
        f_mult, a_mult = surface_fn((track.position.x_m, track.position.y_m))
        reference_amp /= a_mult
        dominant_freq /= f_mult

    log_ref_amp = float(np.log10(max(reference_amp, 1e-9)))
    pad = int(round(FEATURE_PAD_S * FS))
    s = max(0, window.start_idx - pad)
    e = min(result.envelope.shape[1], window.end_idx + pad)
    cadence = _cadence_hz(result.envelope[window.best_channel][s:e], FS)

    return np.array([log_ref_amp, dominant_freq,
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
        self.edit_sliders: List[Slider] = []
        self.crossings: List[Crossing] = []

        self.surfaces: List[Surface] = []
        self.ai_surface_aware = True
        self.map_brush = "pavement"
        self.selected_surface: Optional[Surface] = None
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_now: Optional[Tuple[int, int]] = None
        self.log: List[Tuple[str, Tuple[int, int, int]]] = []

        self.clock_s = 0.0
        self.speed_idx = 3
        self.color_idx = 0
        self.kind_counts = {"human": 0, "animal": 0, "vehicle": 0}
        self._last_running = False
        self.start_ticks = pygame.time.get_ticks()

        self.mode = "sim"               # sim | report | history | viewer | edit | map | charts
        self.chart_idx = 0
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
        self.flagged = []
        self.true_intrusions = 0        # crossings by characters you tagged intruder
        self.intrusions_caught = 0      # of those, how many the AI flagged as novel
        self.events: List[dict] = []    # full per-crossing record (for log/metrics/charts)

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

        # session controls (right column, lower area)
        sx = x + 300
        self.btn_save = Button(sx, 470, 150, 28, "Save session")
        self.btn_end = Button(sx, 504, 200, 28, "End + report")
        self.btn_history = Button(sx, 538, 200, 28, "Past sessions")

        # report / history screen buttons
        by = WIN_H - 52
        self.btn_r_save = Button(40, by, 130, 32, "Save report")
        self.btn_r_export = Button(178, by, 150, 32, "Export log CSV")
        self.btn_r_charts = Button(336, by, 156, 32, "Accuracy charts")
        self.btn_r_history = Button(500, by, 150, 32, "Past sessions")
        self.btn_r_resume = Button(658, by, 110, 32, "Resume")
        self.btn_r_quit = Button(776, by, 100, 32, "Quit")
        self.btn_back = Button(40, by, 150, 32, "Back")
        # accuracy charts screen
        self.btn_c_prev = Button(40, by, 50, 32, "<")
        self.btn_c_next = Button(98, by, 50, 32, ">")
        self.btn_c_back = Button(158, by, 150, 32, "Back")

        # character editor screen
        ex = 620
        self.btn_e_type = Button(ex, 120, 320, 28, "Type: Visitor")
        self.sched_bar = pygame.Rect(ex, 208, 24 * 16, 26)
        self.btn_e_home = Button(ex, 246, 78, 24, "Home")
        self.btn_e_office = Button(ex + 84, 246, 78, 24, "Office")
        self.btn_e_night = Button(ex + 168, 246, 78, 24, "Night")
        self.btn_e_all = Button(ex + 252, 246, 58, 24, "All")
        self.btn_e_clear = Button(ex + 316, 246, 72, 24, "Clear")
        self.btn_e_away = Button(ex, 296, 320, 28, "Away (holiday): off")
        self.btn_e_intruder = Button(ex, 334, 340, 28, "Tag as intruder: off")
        self.btn_e_cross = Button(ex, 380, 240, 28, "Make cross now")
        self.btn_e_active = Button(ex, 416, 360, 28, "In scene: off")
        self.btn_e_done = Button(40, by, 150, 32, "Done")
        self.btn_e_remove = Button(200, by, 150, 32, "Remove character")

        # sim-screen: map and surface-awareness
        self.btn_map = Button(x, 380, 220, 28, "Edit map / surfaces")
        self.btn_surf_aware = Button(x, 416, 280, 28, "AI surface-aware: ON")

        # map editor screen
        mx = 620
        self.surf_buttons = {}
        for i, k in enumerate(SURFACE_BRUSHES):
            self.surf_buttons[k] = Button(mx + (i % 2) * 150, 130 + (i // 2) * 36,
                                          140, 28, SURFACE_TYPES[k]["label"])
        self.btn_s_delete = Button(mx, 222, 200, 28, "Delete selected")
        self.btn_s_clear = Button(mx, 258, 200, 28, "Clear all")
        self.btn_m_done = Button(40, by, 150, 32, "Done")

    def _seed_default_cast(self):
        self.add_character("human", name="Person-1", active=True, mass=80,
                           cadence=1.6, step_freq=30, start=(8, 10), end=(14, 16))
        self.add_character("human", name="Person-2", active=True, mass=78,
                           cadence=2.2, step_freq=32, start=(22, 9), end=(17, 15))
        self.add_character("animal", name="Animal-1", active=True, mass=45,
                           cadence=3.0, step_freq=40, start=(11, 11), end=(18, 15))
        self.kind_counts = {"human": 2, "animal": 1, "vehicle": 0}
        self.characters[0].mover = True           # one resident by default
        self.characters[0].schedule = SCHEDULE_PRESETS["home"][:]
        self.select(self.characters[0])

    # -- character management ----------------------------------------------

    def add_character(self, kind, name=None, active=False, mass=None,
                      cadence=None, step_freq=None, start=None, end=None):
        color = CHAR_COLORS[self.color_idx % len(CHAR_COLORS)]
        self.color_idx += 1
        if start is None:
            start = (self.rng.uniform(2, 12), self.rng.uniform(2, 28))
            end = (self.rng.uniform(18, 28), self.rng.uniform(2, 28))
        base = {"human": "Person", "animal": "Animal", "vehicle": "Vehicle"}[kind]
        defaults = {
            "human": dict(mass=75, cadence=1.8, step_freq=30),
            "animal": dict(mass=40, cadence=3.0, step_freq=40),
            "vehicle": dict(mass=1500, cadence=1.8, step_freq=30),
        }[kind]
        ch = Character(
            name=name if name else "(new {0})".format(base.lower()),
            kind=kind, color=color, path_start=start, path_end=end,
            mass_kg=mass if mass is not None else defaults["mass"],
            cadence_hz=cadence if cadence is not None else defaults["cadence"],
            step_freq_hz=step_freq if step_freq is not None else defaults["step_freq"])
        ch.numbered = name is not None
        ch.active = active
        if active:
            ch.schedule_first(self.clock_s, self.rng)
        self.characters.append(ch)
        self.select(ch)

    def _activate(self, ch: Character):
        """Release a character into the scene if it is not already participating."""
        if not ch.active:
            ch.active = True
            ch.schedule_first(self.clock_s, self.rng)

    def _force_cross(self, ch: Character):
        """Make a character enter the scene immediately (e.g. a burglar now)."""
        detected, ps, pe = self._synthesize_and_learn(ch)
        self.crossings.append(Crossing(
            ch, pygame.time.get_ticks(), 1100, self._last_running, detected,
            (float(ps[0]), float(ps[1])), (float(pe[0]), float(pe[1]))))

    def select(self, ch: Optional[Character]):
        self.selected = ch

    def _open_edit(self, ch: Optional[Character]):
        if ch is None:
            return
        self.selected = ch
        x, y, w = 40, 150, 360
        if ch.kind in ("human", "animal"):
            mmax = 150 if ch.kind == "human" else 90
            self.edit_sliders = [
                Slider(x, y, w, 5, mmax, ch.mass_kg, "Mass (kg)", "mass_kg"),
                Slider(x, y + 64, w, 1.0, 4.0, ch.cadence_hz, "Cadence (steps/s)",
                       "cadence_hz", "{:.2f}"),
                Slider(x, y + 128, w, 20, 45, ch.step_freq_hz, "Foot freq (Hz)",
                       "step_freq_hz"),
                Slider(x, y + 192, w, 0.0, 1.0, ch.spontaneity, "Run chance",
                       "spontaneity", "{:.0%}"),
            ]
        else:
            self.edit_sliders = [
                Slider(x, y, w, 400, 3000, ch.mass_kg, "Weight (kg)", "mass_kg"),
                Slider(x, y + 64, w, 8, 20, ch.axle_freq_hz, "Axle freq (Hz)",
                       "axle_freq_hz"),
                Slider(x, y + 128, w, 0.3, 0.8, ch.axle_spacing_s, "Axle gap (s)",
                       "axle_spacing_s", "{:.2f}"),
            ]
        self.mode = "edit"

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
        if not ch.numbered:        # name it now that it is leaving an impact
            base = {"human": "Person", "animal": "Animal",
                    "vehicle": "Vehicle"}[ch.kind]
            self.kind_counts[ch.kind] += 1
            ch.name = "{0}-{1}".format(base, self.kind_counts[ch.kind])
            ch.numbered = True
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
        surface_fn = self._surface_factor if self.surfaces else None
        if ch.kind in ("human", "animal"):
            sim.add_footstep_train(
                feed, start_s=3.0, path_start_xy=ps, path_end_xy=pe,
                num_steps=8 if ch.kind == "animal" else 6,
                cadence_hz=float(ch.cadence_hz * (1.6 if running else 1.0)),
                mass_kg=float(ch.mass_kg * gain),
                freq_hz=float(ch.step_freq_hz * (1.1 if running else 1.0)),
                surface_fn=surface_fn, label=ch.name)
        else:
            half = ch.mass_kg * self.env.coupling_gain
            sim.add_vehicle_pass(
                feed, start_s=3.0, path_start_xy=ps, path_end_xy=pe,
                axle_masses=(half * 0.45, half * 0.55),
                axle_spacing_s=ch.axle_spacing_s, axle_freq_hz=ch.axle_freq_hz,
                surface_fn=surface_fn, label=ch.name)

        result = self.processor.process(feed)
        if not result.windows:
            self._log("{0}  {1} -> too faint to detect".format(
                self._timestr(), ch.name), WARN)
            return False, ps, pe

        window = max(result.windows, key=lambda w: w.peak_v)
        correct = self._surface_factor if (self.ai_surface_aware and self.surfaces) else None
        vector = extract_signature(result, window, self.env, surface_fn=correct)
        obs = self.memory.observe(vector, self.clock_s)
        self.profile_truth.setdefault(obs.profile_id, Counter())[ch.name] += 1
        mid = ((float(ps[0]) + float(pe[0])) / 2, (float(ps[1]) + float(pe[1])) / 2)
        surf = self._surface_at(*mid)
        self.events.append({
            "t": round(self.clock_s, 1), "true": ch.name, "kind": ch.kind,
            "intruder": ch.is_intruder,
            "surface": surf.kind if surf else "soil", "weather": self.env.key,
            "decision": "new" if obs.is_new else "match",
            "profile": obs.profile_id,
            "distance": round(obs.distance, 3) if np.isfinite(obs.distance) else None,
        })
        if len(self.events) > 20000:
            self.events = self.events[-20000:]

        # Consolidate any same-source split and fold the tallies together.
        for removed, kept in self.memory.consolidate():
            if removed in self.profile_truth:
                self.profile_truth.setdefault(kept, Counter()).update(
                    self.profile_truth.pop(removed))

        run_tag = " [RUN]" if running else ""
        if ch.is_intruder:
            self.true_intrusions += 1
        # Flag only signatures that are clearly unlike anything already known,
        # once the regulars are established, so a regular's momentarily odd
        # reading is not mistaken for an intruder.
        warm = (len(self.memory.enrolled_profiles()) >= 2 and self.clock_s > 3600
                and obs.distance > 1.6)
        if obs.is_new and warm:
            self.flagged.append({"time": self._timestr(), "identity": ch.name,
                                 "distance": round(obs.distance, 2),
                                 "intruder": ch.is_intruder})
            if ch.is_intruder:
                self.intrusions_caught += 1
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
            if not ch.active or ch.away:
                continue          # not released into the scene, or on holiday
            fired = 0
            while self.clock_s >= ch.next_cross_s and fired < 3:
                if ch.mover and not self._char_active(ch):
                    ch.next_cross_s = self._next_active_start(ch)   # off-hours
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

    def _char_active(self, ch: Character) -> bool:
        if ch.away:
            return False
        if not ch.mover:
            return True                       # visitors may pass at any hour
        return ch.schedule[int((self.clock_s % 86400) // 3600)]

    def _next_active_start(self, ch: Character) -> float:
        base = int(self.clock_s // 3600)
        for k in range(1, 49):
            if ch.schedule[(base + k) % 24]:
                return (base + k) * 3600.0
        return self.clock_s + 3600.0          # schedule empty: just wait

    # -- surfaces ----------------------------------------------------------

    def _surface_at(self, x: float, y: float) -> Optional[Surface]:
        for surf in reversed(self.surfaces):     # topmost (last placed) wins
            if surf.contains(x, y):
                return surf
        return None

    def _surface_factor(self, xy) -> Tuple[float, float]:
        surf = self._surface_at(xy[0], xy[1])
        if surf is None:
            return (1.0, 1.0)
        t = SURFACE_TYPES[surf.kind]
        return (t["freq"], t["amp"])

    def _px_to_m(self, px, py) -> Tuple[float, float]:
        return ((px - MAP_RECT.x) / SCALE, SITE_M - (py - MAP_RECT.y) / SCALE)

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
            res = any(ch.name == ident and ch.mover for ch in self.characters)
            regulars.append({"id": p.profile_id, "obs": p.count,
                             "identity": ident, "resident": res,
                             "purity": round(pur, 2)})
        regulars.sort(key=lambda r: -r["obs"])
        suspicious = [{"id": p.profile_id, "obs": p.count,
                       "identity": self._identity(p.profile_id)[0]}
                      for p in tentative]

        # detection metrics (precision/recall of intrusion flagging)
        caught = self.intrusions_caught
        false_alarms = max(0, len(self.flagged) - caught)
        prec = caught / (caught + false_alarms) if (caught + false_alarms) else None
        rec = caught / self.true_intrusions if self.true_intrusions else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        # clustering quality
        purities, dom_counts = [], Counter()
        for p in enrolled:
            c = self.profile_truth.get(p.profile_id, Counter())
            tot = sum(c.values())
            if tot:
                dom, n = c.most_common(1)[0]
                purities.append(n / tot)
                dom_counts[dom] += 1
        avg_purity = sum(purities) / len(purities) if purities else None
        splits = sum(1 for v in dom_counts.values() if v > 1)
        collisions = sum(1 for pp in purities if pp < 0.7)

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
            # ground truth (yours) vs what the AI did, for the overlay
            "true_characters": len(self.characters),
            "true_movers": sum(1 for c in self.characters if c.mover),
            "true_intrusions": self.true_intrusions,
            "intrusions_caught": self.intrusions_caught,
            "ai_profiles": len(self.memory.profiles),
            "ai_enrolled": len(enrolled),
            "ai_flags": len(self.flagged),
            "precision": None if prec is None else round(prec, 2),
            "recall": None if rec is None else round(rec, 2),
            "f1": None if f1 is None else round(f1, 2),
            "avg_purity": None if avg_purity is None else round(avg_purity, 2),
            "splits": splits,
            "collisions": collisions,
            "cast": [{"name": c.name, "kind": c.kind, "mover": c.mover,
                      "intruder": c.is_intruder, "mass_kg": round(c.mass_kg, 1),
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

    def export_log(self) -> str:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        name = "eventlog_{0}.csv".format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        cols = ["t", "true", "kind", "intruder", "surface", "weather",
                "decision", "profile", "distance"]
        with open(os.path.join(SESSIONS_DIR, name), "w", encoding="utf-8",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for e in self.events:
                w.writerow(e)
        self._log("exported {0} events to sessions/{1}".format(
            len(self.events), name), GOOD)
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
        if self.mode == "edit":
            self._draw_edit()
            return
        if self.mode == "map":
            self._draw_map_editor()
            return
        if self.mode == "charts":
            self._draw_charts()
            return
        self.screen.fill(BG)
        self._draw_header()
        self._draw_map()
        self._draw_controls()
        self._draw_database()
        self._draw_log()

    def _name_color(self, name):
        for ch in self.characters:
            if ch.name == name:
                return ch.color
        return INK

    def _draw_charts(self):
        self.screen.fill(BG)
        mp = self._mouse_canvas()
        names = sorted({e["true"] for e in self.events})
        self.screen.blit(self.big.render(
            "ACCURACY OVER TIME", True, INK), (40, 24))
        if not names:
            self.screen.blit(self.font.render(
                "No crossings recorded yet. Run the sim, then come back.",
                True, DIM), (40, 80))
            self.btn_c_back.draw(self.screen, self.font, self.btn_c_back.hit(mp))
            return

        name = names[self.chart_idx % len(names)]
        # the AI profile that this character mostly landed in
        dom, dom_n = None, 0
        for pid, c in self.profile_truth.items():
            if c.get(name, 0) > dom_n:
                dom, dom_n = pid, c[name]

        ev = sorted([e for e in self.events if e["true"] == name],
                    key=lambda e: e["t"])
        tmax = max(e["t"] for e in self.events) or 1.0
        n_true = len(ev)

        self.screen.blit(self.font.render(
            "Character: {0}    (real crossings vs what the AI recognised as its "
            "profile {1})".format(name, dom or "-"), True, self._name_color(name)),
            (40, 58))

        plot = pygame.Rect(90, 110, 1020, 540)
        pygame.draw.rect(self.screen, PANEL, plot, border_radius=4)
        # axes
        pygame.draw.line(self.screen, DIM, (plot.x, plot.bottom),
                         (plot.right, plot.bottom), 1)
        pygame.draw.line(self.screen, DIM, (plot.x, plot.y),
                         (plot.x, plot.bottom), 1)

        def px(t, count):
            x = plot.x + (t / tmax) * plot.width
            y = plot.bottom - (count / max(n_true, 1)) * (plot.height - 10)
            return (int(x), int(y))

        truth_pts, ai_pts = [px(0, 0)], [px(0, 0)]
        tc = ac = 0
        for e in ev:
            tc += 1
            if e["profile"] == dom:
                ac += 1
            truth_pts.append(px(e["t"], tc))
            ai_pts.append(px(e["t"], ac))
        if len(truth_pts) > 1:
            pygame.draw.lines(self.screen, ACCENT, False, truth_pts, 2)
            pygame.draw.lines(self.screen, GOOD, False, ai_pts, 2)

        # axis labels
        for frac in (0.0, 0.5, 1.0):
            tx = plot.x + frac * plot.width
            self.screen.blit(self.small.render(
                "{0:.1f}d".format((tmax * frac) / 86400), True, DIM),
                (int(tx) - 10, plot.bottom + 6))
        self.screen.blit(self.small.render(str(n_true), True, DIM),
                         (plot.x - 30, plot.y))
        self.screen.blit(self.small.render("0", True, DIM),
                         (plot.x - 16, plot.bottom - 8))

        # legend + stats
        pygame.draw.line(self.screen, ACCENT, (plot.right - 320, plot.y + 16),
                         (plot.right - 290, plot.y + 16), 3)
        self.screen.blit(self.small.render("real crossings", True, INK),
                         (plot.right - 282, plot.y + 9))
        pygame.draw.line(self.screen, GOOD, (plot.right - 320, plot.y + 36),
                         (plot.right - 290, plot.y + 36), 3)
        self.screen.blit(self.small.render("recognised by AI", True, INK),
                         (plot.right - 282, plot.y + 29))

        acc = ac / n_true if n_true else 0.0
        last_truth = ev[-1]["t"]
        ai_times = [e["t"] for e in ev if e["profile"] == dom]
        msg = "recognised {0}/{1} ({2:.0%}).".format(ac, n_true, acc)
        if not ai_times:
            msg += "  Never settled into one profile."
        elif last_truth - max(ai_times) > 0.05 * tmax:
            msg += "  CUT OFF: recognition stopped at {0:.1f}d, crossings " \
                   "continued to {1:.1f}d.".format(
                       max(ai_times) / 86400, last_truth / 86400)
        self.screen.blit(self.font.render(msg, True, WARN), (90, plot.bottom + 26))
        self.screen.blit(self.small.render(
            "Lines together = accurate. AI line flat while real line climbs = the "
            "AI lost this identity (split or mislabelled).", True, DIM),
            (90, plot.bottom + 48))

        self.btn_c_prev.draw(self.screen, self.font, self.btn_c_prev.hit(mp))
        self.btn_c_next.draw(self.screen, self.font, self.btn_c_next.hit(mp))
        self.btn_c_back.draw(self.screen, self.font, self.btn_c_back.hit(mp))

    def _draw_map_editor(self):
        self.screen.fill(BG)
        mp = self._mouse_canvas()
        ground, grid = ENV_VISUALS.get(self.env.key, ((36, 72, 44), (52, 92, 60)))
        pygame.draw.rect(self.screen, ground, MAP_RECT, border_radius=6)
        prev = self.screen.get_clip()
        self.screen.set_clip(MAP_RECT)
        for g in range(0, 31, 5):
            pygame.draw.line(self.screen, grid, m_to_px(g, 0), m_to_px(g, 30), 1)
            pygame.draw.line(self.screen, grid, m_to_px(0, g), m_to_px(30, g), 1)
        self._draw_surfaces(editing=True)
        if self._drag_start and self._drag_now:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_now
            pr = pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            pygame.draw.rect(self.screen, SURFACE_TYPES[self.map_brush]["color"], pr, 2)
        self.screen.set_clip(prev)
        for i, (sx, sy) in enumerate(SENSOR_POSITIONS_M):
            p = m_to_px(sx, sy)
            pygame.draw.rect(self.screen, ACCENT,
                             pygame.Rect(p[0] - 6, p[1] - 6, 12, 12), border_radius=2)
        self.screen.blit(self.big.render("MAP EDITOR", True, INK), (MAP_RECT.x, 22))

        mx = 620
        self.screen.blit(self.font.render(
            "Surface brush, then drag on the map to paint a zone:", True, DIM),
            (mx, 100))
        for k, btn in self.surf_buttons.items():
            btn.draw(self.screen, self.font, self.map_brush == k or btn.hit(mp))
        self.btn_s_delete.draw(self.screen, self.font, self.btn_s_delete.hit(mp))
        self.btn_s_clear.draw(self.screen, self.font, self.btn_s_clear.hit(mp))
        self.screen.blit(self.small.render(
            "Drag to paint. Click a zone to select it, then Delete.", True, DIM),
            (mx, 300))
        self.screen.blit(self.small.render(
            "Brush: {0}  (harder surface = higher freq + stronger signal)".format(
                SURFACE_TYPES[self.map_brush]["label"]), True, GOLD), (mx, 322))
        self.screen.blit(self.small.render(
            "Surface effects are estimates, like the weather values.", True, DIM),
            (mx, 344))
        self.btn_m_done.draw(self.screen, self.font, self.btn_m_done.hit(mp))

    def _draw_edit(self):
        self.screen.fill(BG)
        ch = self.selected
        if ch is None:
            self.mode = "sim"
            return
        mp = self._mouse_canvas()
        self.screen.blit(self.big.render(
            "EDIT: {0}  ({1})".format(ch.name, ch.kind), True, ch.color), (40, 24))
        if ch.active:
            self.screen.blit(self.font.render(
                "In the scene: producing crossings.", True, GOOD), (40, 56))
        else:
            self.screen.blit(self.font.render(
                "NOT in the scene yet. Pick a schedule on the right (or 'In scene' / "
                "'Make cross now') to make it appear.", True, WARN), (40, 56))

        # left: biometrics
        self.screen.blit(self.font.render("Biometrics", True, DIM), (40, 110))
        for s in self.edit_sliders:
            s.draw(self.screen, self.font)

        # right: schedule and behaviour
        ex = 620
        self.btn_e_type.label = ("Type: Scheduled mover" if ch.mover
                                 else "Type: Visitor (occasional)")
        self.btn_e_type.draw(self.screen, self.font, ch.mover or self.btn_e_type.hit(mp))

        if ch.mover:
            self.screen.blit(self.small.render(
                "Active hours (click to toggle; leave gaps for breaks)", True, DIM),
                (ex, 188))
            cell = self.sched_bar.width // 24
            for h in range(24):
                cr = pygame.Rect(self.sched_bar.x + h * cell, self.sched_bar.y,
                                 cell - 1, self.sched_bar.height)
                pygame.draw.rect(self.screen, GOOD if ch.schedule[h] else (52, 56, 68),
                                 cr)
                if h % 3 == 0:
                    self.screen.blit(self.small.render(str(h), True, DIM),
                                     (cr.x, self.sched_bar.bottom + 3))
            for b in (self.btn_e_home, self.btn_e_office, self.btn_e_night,
                      self.btn_e_all, self.btn_e_clear):
                b.draw(self.screen, self.small, b.hit(mp))
        else:
            self.screen.blit(self.small.render(
                "Visitor: appears occasionally at any hour.", True, DIM), (ex, 200))

        self.btn_e_away.label = "Away (holiday): {0}".format("ON" if ch.away else "off")
        self.btn_e_away.draw(self.screen, self.font, ch.away or self.btn_e_away.hit(mp))
        self.btn_e_intruder.label = "Tag as intruder: {0}".format(
            "ON" if ch.is_intruder else "off")
        self.btn_e_intruder.draw(self.screen, self.font,
                                 ch.is_intruder or self.btn_e_intruder.hit(mp))
        self.btn_e_cross.draw(self.screen, self.font, self.btn_e_cross.hit(mp))
        self.btn_e_active.label = "In scene (produces crossings): {0}".format(
            "ON" if ch.active else "off")
        self.btn_e_active.draw(self.screen, self.font,
                               ch.active or self.btn_e_active.hit(mp))

        self.screen.blit(self.small.render(
            "The AI is never told any of this. It only ever receives the seismic feed.",
            True, DIM), (ex, 456))

        self.btn_e_done.draw(self.screen, self.font, self.btn_e_done.hit(mp))
        self.btn_e_remove.draw(self.screen, self.font, self.btn_e_remove.hit(mp))

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

        # overlay: your ground truth vs what the AI did
        ti = rep.get("true_intrusions", 0)
        caught = rep.get("intrusions_caught", 0)
        false_alarms = max(0, rep.get("ai_flags", 0) - caught)
        self.screen.blit(self.font.render(
            "YOU set up {0} characters ({1} scheduled) and triggered {2} "
            "intrusions.".format(rep.get("true_characters", "?"),
                                 rep.get("true_movers", "?"), ti),
            True, ACCENT), (40, 76))
        self.screen.blit(self.font.render(
            "AI built {0} confirmed profiles ({1} total) and caught {2}/{3} of "
            "your intrusions  ({4} false alarms).".format(
                rep.get("ai_enrolled", "?"), rep.get("ai_profiles", "?"),
                caught, ti, false_alarms),
            True, ACCENT), (40, 96))

        def _pct(v):
            return "n/a" if v is None else "{0:.0%}".format(v)
        self.screen.blit(self.font.render(
            "metrics: intrusion precision {0}  recall {1}  F1 {2}   |   "
            "avg purity {3}   |   splits {4}  collisions {5}".format(
                _pct(rep.get("precision")), _pct(rep.get("recall")),
                _pct(rep.get("f1")), _pct(rep.get("avg_purity")),
                rep.get("splits", 0), rep.get("collisions", 0)),
            True, GOOD), (40, 114))

        # left: the learned database (regulars)
        self.screen.blit(self.big.render(
            "Known regulars the system learned", True, GOOD), (40, 132))
        y = 164
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
            "Flagged / one-off activity", True, WARN), (rx, 132))
        self.screen.blit(self.small.render(
            "novel signatures after the regulars were known, and one-offs that "
            "never recurred", True, DIM), (rx, 158))
        y = 180
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
            for b in (self.btn_r_save, self.btn_r_export, self.btn_r_charts,
                      self.btn_r_history, self.btn_r_resume, self.btn_r_quit):
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
            "sim {0}{1}  ({2:.1f} days elapsed)    speed {3}    learned {4}    "
            "real run {5}".format(
                self._timestr(), night, self.clock_s / 86400.0,
                SPEED_LABELS[self.speed_idx], learned, realstr),
            True, DIM), (370, 22))
        self.screen.blit(self.small.render(
            "F=fullscreen  ESC=quit", True, DIM), (WIN_W - 190, 24))

    def _surf_rect(self, surf: Surface) -> pygame.Rect:
        sx = MAP_RECT.x + surf.x * SCALE
        sy = MAP_RECT.y + (SITE_M - (surf.y + surf.h)) * SCALE
        return pygame.Rect(int(sx), int(sy), int(surf.w * SCALE), int(surf.h * SCALE))

    def _draw_surfaces(self, editing=False):
        for surf in self.surfaces:
            rect = self._surf_rect(surf)
            pygame.draw.rect(self.screen, SURFACE_TYPES[surf.kind]["color"], rect)
            sel = editing and surf is self.selected_surface
            pygame.draw.rect(self.screen, (255, 255, 255) if sel else (24, 26, 32),
                             rect, 2 if sel else 1)

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
        self._draw_surfaces()
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
            if c.character.is_intruder:
                pygame.draw.circle(self.screen, BAD, p, 15, 2)
            if c.character.mover:
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

        # add character
        for b in (self.btn_human, self.btn_animal, self.btn_vehicle):
            b.draw(self.screen, self.font, b.hit(mp))

        pygame.draw.line(self.screen, GRID, (x, 292), (WIN_W - 20, 292), 1)

        # editing is on its own screen
        self.screen.blit(self.font.render(
            "Click a character in the roster to open its", True, DIM), (x, 312))
        self.screen.blit(self.font.render(
            "editor (biometrics, schedule, holiday, intruder).", True, DIM),
            (x, 334))
        self.btn_map.draw(self.screen, self.font, self.btn_map.hit(mp))
        self.btn_surf_aware.label = "AI surface-aware: {0}".format(
            "ON" if self.ai_surface_aware else "OFF")
        self.btn_surf_aware.draw(self.screen, self.font,
                                 self.ai_surface_aware or self.btn_surf_aware.hit(mp))

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
            tag, tcol = "", DIM
            if ch.away:
                tag, tcol = "away", DIM
            elif ch.is_intruder:
                tag, tcol = "intruder", BAD
            elif ch.mover:
                tag, tcol = "scheduled", GOLD
            if tag:
                self.screen.blit(self.small.render(tag, True, tcol),
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
        caught = self.intrusions_caught
        fa = max(0, len(self.flagged) - caught)
        self.screen.blit(self.font.render(
            "enrolled {0}  tentative {1}   |   intrusions {2}/{3}  false {4}".format(
                len(enrolled), len(tentative), caught, self.true_intrusions, fa),
            True, DIM), (x + 250, BOTTOM_Y + 12))

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
        if self.mode == "edit":
            return self._handle_edit(event)
        if self.mode == "map":
            return self._handle_map(event)
        if self.mode == "charts":
            return self._handle_charts(event)
        return self._handle_sim(event)

    def _handle_map(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.mode = "sim"
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            p = event.pos
            for k, btn in self.surf_buttons.items():
                if btn.hit(p):
                    self.map_brush = k
                    return True
            if self.btn_s_delete.hit(p):
                if self.selected_surface in self.surfaces:
                    self.surfaces.remove(self.selected_surface)
                self.selected_surface = None
                return True
            if self.btn_s_clear.hit(p):
                self.surfaces = []
                self.selected_surface = None
                return True
            if self.btn_m_done.hit(p):
                self.mode = "sim"
                return True
            if MAP_RECT.collidepoint(p):
                self._drag_start = p
                self._drag_now = p
            return True
        if event.type == pygame.MOUSEMOTION and self._drag_start:
            self._drag_now = event.pos
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._drag_start:
            x0, y0 = self._drag_start
            x1, y1 = event.pos
            self._drag_start = None
            self._drag_now = None
            if abs(x1 - x0) > 6 and abs(y1 - y0) > 6:
                a = self._px_to_m(x0, y0)
                b = self._px_to_m(x1, y1)
                self.surfaces.append(Surface(
                    self.map_brush, min(a[0], b[0]), min(a[1], b[1]),
                    abs(a[0] - b[0]), abs(a[1] - b[1])))
            else:
                cx, cy = self._px_to_m(x1, y1)
                self.selected_surface = self._surface_at(cx, cy)
            return True
        return True

    def _handle_edit(self, event):
        ch = self.selected
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.mode = "sim"
            return True
        for s in self.edit_sliders:
            if s.handle(event) and ch is not None:
                setattr(ch, s.attr, s.value)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and ch is not None:
            p = event.pos
            if self.btn_e_type.hit(p):
                ch.mover = not ch.mover
                if ch.mover:
                    if not any(ch.schedule):
                        ch.schedule = SCHEDULE_PRESETS["home"][:]
                    self._activate(ch)
            elif self.btn_e_home.hit(p):
                ch.mover = True; ch.schedule = SCHEDULE_PRESETS["home"][:]
                self._activate(ch)
            elif self.btn_e_office.hit(p):
                ch.mover = True; ch.schedule = SCHEDULE_PRESETS["office"][:]
                self._activate(ch)
            elif self.btn_e_night.hit(p):
                ch.mover = True; ch.schedule = SCHEDULE_PRESETS["night"][:]
                self._activate(ch)
            elif self.btn_e_all.hit(p):
                ch.mover = True; ch.schedule = SCHEDULE_PRESETS["all"][:]
                self._activate(ch)
            elif self.btn_e_clear.hit(p):
                ch.schedule = [False] * 24
            elif self.btn_e_away.hit(p):
                ch.away = not ch.away
                if not ch.away:
                    ch.next_cross_s = self.clock_s + 1.0
            elif self.btn_e_intruder.hit(p):
                ch.is_intruder = not ch.is_intruder
            elif self.btn_e_cross.hit(p):
                self._force_cross(ch)
            elif self.btn_e_active.hit(p):
                ch.active = not ch.active
                if ch.active:
                    ch.schedule_first(self.clock_s, self.rng)
            elif self.btn_e_done.hit(p):
                self.mode = "sim"
            elif self.btn_e_remove.hit(p):
                self.characters.remove(ch)
                self.select(self.characters[0] if self.characters else None)
                self.mode = "sim"
            elif ch.mover and self.sched_bar.collidepoint(p):
                h = int((p[0] - self.sched_bar.x) / (self.sched_bar.width // 24))
                if 0 <= h < 24:
                    ch.schedule[h] = not ch.schedule[h]
                    self._activate(ch)
        return True

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
            elif self.btn_r_export.hit(p):
                self.export_log()
            elif self.btn_r_charts.hit(p):
                self.mode = "charts"
            elif self.btn_r_resume.hit(p):
                self.mode = "sim"
            elif self.btn_r_history.hit(p):
                self._enter_history()
            elif self.btn_r_quit.hit(p):
                self._autosave()
                return False
        return True

    def _handle_charts(self, event):
        names = sorted({e["true"] for e in self.events})
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.mode = "report"
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.btn_c_back.hit(event.pos):
                self.mode = "report"
            elif self.btn_c_prev.hit(event.pos) and names:
                self.chart_idx = (self.chart_idx - 1) % len(names)
            elif self.btn_c_next.hit(event.pos) and names:
                self.chart_idx = (self.chart_idx + 1) % len(names)
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
        if self.btn_map.hit(pos):
            self.mode = "map"
            return
        if self.btn_surf_aware.hit(pos):
            self.ai_surface_aware = not self.ai_surface_aware
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
            self.add_character("human"); self._open_edit(self.selected); return
        if self.btn_animal.hit(pos):
            self.add_character("animal"); self._open_edit(self.selected); return
        if self.btn_vehicle.hit(pos):
            self.add_character("vehicle"); self._open_edit(self.selected); return
        rx = PANEL_X + 300
        for i, ch in enumerate(self.characters):
            r = pygame.Rect(rx, 330 + i * 24, 280, 22)
            if r.collidepoint(pos):
                self._open_edit(ch)
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

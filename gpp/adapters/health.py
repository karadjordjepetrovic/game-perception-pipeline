"""Hits and health from CoD's damage cues: the red tint at the screen edges and the red
hit-direction arcs around the crosshair.

Per frame (downscaled to 480x270, HSV), the share of strongly red pixels in four regions:
- edge:    a ring around the screen edges (the damage tint)
- arc:     an annulus around the crosshair, 0.10-0.32 of the frame height (hit-direction arcs)
- core:    the crosshair area inside it; ignored, because a red-dot sight is red all the time
- rest:    everything else; red here means a red scene (fire, red walls), so it is subtracted
           from edge and arc, and a red room doesn't read as damage

    edge_score = max(0, edge - rest - edge_noise) / full_edge
    arc_score  = max(0, arc  - rest - arc_noise)  / full_arc
    damage     = clamp(max(edge_score, arc_score)),  hit = damage > 0

health drops at once by the damage and recovers slowly (recover_s), like the game's own
regeneration. The agent goes to RECOVER on any hit (gpp/agent/fsm.py).

Calibration (bots clip, 1612 sampled frames): rest p50 0.0025; edge p99 0.012; arc p99 0.008.
With these noise floors, 6 hit events in 28 frames, at moments with visible red cues.
"""
import math

import cv2
import numpy as np

from ..stage import Stage


class HealthStage(Stage):
    name = "health"

    def __init__(self, full_damage=0.06, full_arc=0.05, edge_noise=0.004, arc_noise=0.006,
                 inner=(0.15, 0.15, 0.85, 0.85), arc_radii=(0.10, 0.32), recover_s=1.5, every_n=2):
        self.full_damage = full_damage  # edge red share (above rest) that counts as no health left
        self.full_arc = full_arc
        self.edge_noise = edge_noise
        self.arc_noise = arc_noise
        self.inner = inner              # the edge ring is everything outside this box (fractions)
        self.arc_radii = arc_radii      # annulus around the crosshair, in frame heights
        self.recover_s = recover_s
        self.every_n = every_n

    def setup(self, info):
        self.size = (480, 270)
        w, h = self.size
        x1, y1, x2, y2 = self.inner
        self.edge = np.ones((h, w), bool)
        self.edge[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = False
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(xx - w / 2, yy - h / 2) / h
        self.arc = (r > self.arc_radii[0]) & (r < self.arc_radii[1]) & ~self.edge
        core = r <= self.arc_radii[0]
        self.rest = ~self.edge & ~self.arc & ~core
        self.dt = self.every_n / info.fps
        self.health = 1.0

    def process(self, index, frame):
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        red = ((hue < 10) | (hue > 170)) & (sat > 100) & (val > 50)
        edge, arc, rest = float(red[self.edge].mean()), float(red[self.arc].mean()), float(red[self.rest].mean())
        edge_score = max(0.0, edge - rest - self.edge_noise) / self.full_damage
        arc_score = max(0.0, arc - rest - self.arc_noise) / self.full_arc
        damage = min(1.0, max(edge_score, arc_score))
        now = 1.0 - damage
        if now < self.health:
            self.health = now
        else:
            self.health += (now - self.health) * (1.0 - math.exp(-self.dt / self.recover_s))
        return {"health": self.health, "damage": damage, "hit": damage > 0.0,
                "red_edge": edge, "red_arc": arc, "red_rest": rest}

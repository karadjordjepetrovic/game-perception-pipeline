"""Cover from the depth map, and where to hide behind it.

Finding cover (find_cover): in a horizontal band around eye height (38-62% of the frame
height), take each depth column's median. Anything close there is an obstacle at head and
torso height (a crate, a wall corner, a car); the floor in that band is 10+ m away, so it
doesn't count. Columns covered by a person's box are skipped (an NPC isn't cover).
Neighbouring close columns form one cover object: its left/right edge angles and distance.

Distance is metric when gpp.world has calibrated the depth scale from people's box
heights (dist = k / depth). Before that, columns among the closest 20% of the frame count
as cover at an assumed distance.

Choosing where to hide (choose_hide): top-down view, the player at (0, 0) looking along +z,
x to the right. For each cover object C and the nearest threat T, the hiding spot is
H = C + offset * (C - T) / |C - T|: just behind the object, on the side away from the
threat. The spot is checked against every threat (is the object between H and it?), and
the cheapest one wins: walking distance + a penalty per threat it doesn't block + a
penalty for cover right next to a threat. Objects narrower than min_width_m (a pole, a
lamp post) are skipped: too thin to hide behind.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Cover:
    yaw: float                       # centre angle from the crosshair, degrees (+ = right)
    yaw_left: float
    yaw_right: float
    dist_m: float
    metric: bool                     # dist from the calibrated depth scale (else assumed)

    @property
    def pos(self):
        return _pos(self.yaw, self.dist_m)

    @property
    def half_width_m(self):
        return self.dist_m * math.tan(math.radians(abs(self.yaw_right - self.yaw_left)) / 2)

    def to_dict(self):
        return {"yaw": round(self.yaw, 1), "left": round(self.yaw_left, 1), "right": round(self.yaw_right, 1),
                "dist_m": round(self.dist_m, 1), "metric": self.metric}


@dataclass
class HideChoice:
    cover: Cover
    hide: tuple                      # (x, z) metres from the player, x right, z forward
    covered: int                     # threats the cover blocks from the hiding spot
    threats: int

    @property
    def dist_m(self):
        return math.hypot(*self.hide)


def _pos(yaw_deg, dist):
    a = math.radians(yaw_deg)
    return dist * math.sin(a), dist * math.cos(a)


def find_cover(depth, width, height, fx, depth_scale=None, exclude_boxes=(), band=(0.38, 0.62),
               max_m=8.0, min_width_deg=3.0, assumed_m=3.0, closest_share=0.2, stride=2):
    d = np.asarray(depth, dtype=np.float32)
    dh, dw = d.shape[:2]
    cols = np.median(d[int(band[0] * dh):int(band[1] * dh):stride, ::stride], axis=0)
    xs = (np.arange(len(cols)) * stride + stride / 2) * width / dw  # column centre in frame pixels

    if depth_scale:
        dist = np.where(cols > 0, depth_scale / np.maximum(cols, 1e-6), np.inf)
        close = dist <= max_m
        metric = True
    else:
        close = cols >= np.quantile(cols, 1.0 - closest_share)
        dist = np.full_like(cols, assumed_m)
        metric = False

    for x1, y1, x2, y2 in exclude_boxes:
        pad = 0.1 * (x2 - x1)
        close &= ~((xs >= x1 - pad) & (xs <= x2 + pad))

    covers = []
    j, n = 0, len(cols)
    while j < n:
        if not close[j]:
            j += 1
            continue
        k = j
        while k + 1 < n and close[k + 1]:
            k += 1
        left = math.degrees(math.atan((xs[j] - width / 2) / fx))
        right = math.degrees(math.atan((xs[k] - width / 2) / fx))
        if right - left >= min_width_deg:
            centre = math.degrees(math.atan(((xs[j] + xs[k]) / 2 - width / 2) / fx))
            covers.append(Cover(centre, left, right, float(np.median(dist[j:k + 1])), metric))
        j = k + 1
    return covers


def _blocks(cover, hide, threat):
    """Is the cover object between the hiding spot and the threat (top-down, as a segment)?"""
    cx, cz = cover.pos
    hx, hz = hide
    tx, tz = threat
    to_t, to_c = math.atan2(tx - hx, tz - hz), math.atan2(cx - hx, cz - hz)
    d_c = math.hypot(cx - hx, cz - hz)
    if d_c >= math.hypot(tx - hx, tz - hz):
        return False                 # threat is closer than the cover
    half = math.atan2(cover.half_width_m, max(d_c, 1e-3))
    diff = abs((to_t - to_c + math.pi) % (2 * math.pi) - math.pi)
    return diff <= half


def choose_hide(covers, threats, offset_m=1.0, uncovered_penalty=10.0, near_threat_m=2.0,
                min_width_m=0.6) -> Optional[HideChoice]:
    """threats: list of (yaw_deg, dist_m). Returns the best hiding spot, or None without cover."""
    if not covers:
        return None
    pts = [_pos(y, d) for y, d in threats]
    best, best_cost = None, math.inf
    for c in covers:
        if 2 * c.half_width_m < min_width_m:
            continue
        cx, cz = c.pos
        if pts:
            tx, tz = min(pts, key=lambda p: math.hypot(p[0], p[1]))
            vx, vz = cx - tx, cz - tz
            n = math.hypot(vx, vz)
            if n < 1e-6:
                continue
            hide = (cx + offset_m * vx / n, cz + offset_m * vz / n)
        else:                        # hit from an unknown direction: get next to the nearest object
            n = math.hypot(cx, cz)
            hide = (cx * (1 - offset_m / n), cz * (1 - offset_m / n)) if n > offset_m else (0.0, 0.0)
        covered = sum(_blocks(c, hide, p) for p in pts)
        cost = math.hypot(*hide) + uncovered_penalty * (len(pts) - covered)
        if any(math.hypot(cx - px, cz - pz) < near_threat_m for px, pz in pts):
            cost += uncovered_penalty
        if cost < best_cost:
            best, best_cost = HideChoice(c, hide, covered, len(pts)), cost
    return best

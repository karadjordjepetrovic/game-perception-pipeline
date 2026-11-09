"""WorldState: what the decision agent knows, built from the latest stage results.

The agent never reads raw detections. This module turns them into stable facts:

- targets:  tracked people with screen-space aim error in degrees, a distance estimate,
            a range class (near / mid / far, with hysteresis) and how long they've been seen
- memory:   targets that just disappeared are kept for a short time (visible=False), so the
            agent can search for them instead of forgetting them on the first missed frame.
            With the motion stage (gpp/adapters/motion.py), remembered targets are moved by
            the camera's turn each frame, so their screen position and angle stay on the
            spot where the NPC was, instead of freezing on the screen while the view turns.
            For up to predict_s they also keep moving the way the NPC was last moving.
- motion:   each visible target's angular velocity in the world (deg/s): its change in screen
            angle plus the camera's turn, smoothed. Used to carry lost targets forward and
            for aim lead
- self:     ammo from the HUD text, and how blocked the path ahead looks from the depth map

Angles: pinhole camera with the game's horizontal FOV. fx = (W/2) / tan(hfov/2), square
pixels, so fy = fx. yaw_err > 0 means the target is right of the crosshair, pitch_err > 0
means above it.

Distance, two estimates fused:
- box height: a person is ~1.8 m tall, so dist = height_m * fy / box_h. Metric, but a
  crouching or half-hidden target has a short box and reads as further away than it is.
- depth: Depth Anything V2 gives relative inverse depth (bigger = closer, no unit), but
  occlusion and pose don't change it. Inverse depth ~ k / dist, and k is calibrated from the
  box-height distance of targets that look like a whole standing person (box at least
  2x taller than wide): k = depth_rel * dist_box, median over them, smoothed over frames.
  Then every target, crouched or not, gets dist_depth = k / depth_rel.
distance_m is dist_depth once k is calibrated, dist_box before that or without depth.
Assumes the depth model's shift term is ~0; good enough to rank targets and sort near/far.
"""
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .cover import find_cover

NEAR, MID, FAR = "near", "mid", "far"
_AMMO_RE = re.compile(r"^\s*(\d{1,3})\s*(?:[/|]\s*(\d{1,4}))?\s*$")


@dataclass
class WorldConfig:
    hfov_deg: float = 80.0           # in-game horizontal FOV (CoD default 80)
    person_height_m: float = 1.8
    near_m: float = 10.0             # closer than this: near
    far_m: float = 30.0              # further than this: far
    range_margin: float = 0.15       # hysteresis: leave a range class only 15% past its threshold
    min_frames_seen: int = 3         # a track must be seen this many frames before it counts
    memory_s: float = 2.0            # keep a lost target this long
    aim_from_top: float = 0.2        # aim point: 20% down from the box top (upper chest)
    min_conf: float = 0.25           # ByteTrack already filters; a higher cut here breaks tracks into pieces
    min_box_h_px: int = 12
    max_box_h_frac: float = 0.9      # taller boxes are usually the player's own arms/weapon
    assume_foe: bool = True          # no friend-or-foe stage yet: unknown counts as foe
    ammo_region: tuple = (0.6, 0.6, 1.0, 1.0)  # HUD ammo search area, fractions (x1, y1, x2, y2)
    ammo_confirm: int = 2            # same value in this many OCR results in a row before it counts
    ammo_expiry_s: float = 3.0       # forget the ammo count if it isn't read again for this long
    use_depth_distance: bool = True  # distance from calibrated depth when possible (see module doc)
    standing_aspect: float = 2.0     # box h/w at least this: whole standing person, used to calibrate depth
    depth_scale_alpha: float = 0.1   # smoothing of the depth scale over frames
    rate_alpha: float = 0.3          # smoothing of each target's angular velocity
    max_rate_dps: float = 120.0      # clamp: faster readings are box jitter, not movement
    min_rate_samples: int = 3        # velocity is trusted after this many consecutive frames
    predict_s: float = 0.5           # carry a lost target's movement forward this long
    cover_max_m: float = 8.0         # cover: obstacles at eye height closer than this (gpp/cover.py)
    cover_min_width_deg: float = 3.0


@dataclass
class Target:
    id: int
    box: tuple
    aim: tuple                       # (x, y) pixels
    yaw_err: float                   # degrees, + = right
    pitch_err: float                 # degrees, + = up
    width_deg: float                 # angular width of the box
    distance_m: float                # the one the agent uses
    range: str
    conf: float
    first_seen: float                # seconds
    last_seen: float
    frames_seen: int = 1
    visible: bool = True
    foe: Optional[bool] = None       # None = unknown
    depth_rel: Optional[float] = None
    dist_box_m: Optional[float] = None
    dist_depth_m: Optional[float] = None
    yaw_rate: float = 0.0            # world angular velocity, deg/s (+ = moving right)
    pitch_rate: float = 0.0          # deg/s (+ = moving up)
    rate_samples: int = 0            # consecutive frames the velocity is built from

    def to_dict(self):
        r = lambda v: round(v, 1) if v is not None else None
        return {"id": self.id, "yaw": round(self.yaw_err, 2), "pitch": round(self.pitch_err, 2),
                "aim_px": [round(self.aim[0]), round(self.aim[1])], "dist_m": r(self.distance_m),
                "dist_box": r(self.dist_box_m), "dist_depth": r(self.dist_depth_m),
                "range": self.range, "visible": self.visible,
                "rate": [round(self.yaw_rate, 1), round(self.pitch_rate, 1)]}


@dataclass
class SelfState:
    ammo: Optional[int] = None       # rounds in the magazine, None = not read
    reserve: Optional[int] = None
    health: Optional[float] = None   # 0..1, None = not read (no health stage)
    last_hit: Optional[float] = None # time of the last hit (seconds), None = no hit yet
    blocked: dict = field(default_factory=dict)  # {"left": 0..1, "centre": .., "right": ..}


@dataclass
class WorldState:
    index: int
    t: float
    width: int
    height: int
    targets: list                    # confirmed targets, visible and remembered
    me: SelfState
    camera: Optional[tuple] = None   # (yaw, pitch) the view turned this frame, degrees; None = unknown
    has_motion: bool = False         # a motion stage is running: remembered targets follow the camera
    covers: list = field(default_factory=list)  # gpp.cover.Cover objects in view
    fx: float = 0.0                  # focal length in pixels (pinhole camera)

    def visible_foes(self, assume_foe=True):
        return [t for t in self.targets if t.visible and (t.foe if t.foe is not None else assume_foe)]

    def lost_foes(self, assume_foe=True):
        return [t for t in self.targets if not t.visible and (t.foe if t.foe is not None else assume_foe)]


def range_class(distance, previous, cfg):
    """near / mid / far with hysteresis around the thresholds."""
    m = cfg.range_margin
    if previous == NEAR:
        if distance <= cfg.near_m * (1 + m):
            return NEAR
    elif previous == FAR:
        if distance >= cfg.far_m * (1 - m):
            return FAR
    elif previous == MID:
        if cfg.near_m * (1 - m) < distance < cfg.far_m * (1 + m):
            return MID
    if distance < cfg.near_m:
        return NEAR
    if distance > cfg.far_m:
        return FAR
    return MID


def parse_ammo(text_results, width, height, region):
    """Magazine (and reserve) count from HUD text in the ammo region: '30', '30/120', '30 | 120'.

    Heuristic: of the matching HUD strings in the region, the tallest box wins (the
    magazine count is the biggest number there). Returns (ammo, reserve) or (None, None).
    """
    x1, y1, x2, y2 = region[0] * width, region[1] * height, region[2] * width, region[3] * height
    best = None
    for det in text_results or []:
        if det.get("label") != "hud":
            continue
        bx1, by1, bx2, by2 = det["box"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        match = _AMMO_RE.match(det.get("text", ""))
        if not match:
            continue
        h = by2 - by1
        if best is None or h > best[0]:
            best = (h, int(match.group(1)), int(match.group(2)) if match.group(2) else None)
    return (best[1], best[2]) if best else (None, None)


def blocked_ahead(depth, stride=4):
    """How close the scene is in the lower-centre band, left / centre / right, 0..1.

    depth is relative inverse depth (bigger = closer). Each value is the band's median,
    placed between the frame's 5th and 95th percentile: ~1 means as close as the closest
    things in view (a wall right ahead), ~0 means as far as the furthest.
    """
    d = np.asarray(depth, dtype=np.float32)[::stride, ::stride]  # 518x518 -> 130x130: same answer, ~16x cheaper
    h, w = d.shape[:2]
    lo, hi = np.percentile(d, (5, 95))
    if hi <= lo:
        return {"left": 0.0, "centre": 0.0, "right": 0.0}
    band = d[int(h * 0.35):int(h * 0.65)]
    thirds = {"left": band[:, int(w * 0.15):int(w * 0.38)],
              "centre": band[:, int(w * 0.38):int(w * 0.62)],
              "right": band[:, int(w * 0.62):int(w * 0.85)]}
    return {k: float(np.clip((np.median(v) - lo) / (hi - lo), 0.0, 1.0)) for k, v in thirds.items()}


class WorldBuilder:
    """Keeps track memory between frames and builds a WorldState from each frame's results."""

    def __init__(self, width, height, fps, cfg=None):
        self.cfg = cfg or WorldConfig()
        self.width, self.height, self.fps = width, height, fps
        self.fx = (width / 2) / math.tan(math.radians(self.cfg.hfov_deg) / 2)
        self.tracks = {}             # id -> Target (seen this frame or remembered)
        self.ammo = (None, None)
        self.ammo_t = None           # when the confirmed count was last read
        self.ammo_candidate = (None, 0)  # (reading, times in a row)
        self.last_text = None
        self.depth_scale = None      # k in depth_rel = k / dist, calibrated from standing targets
        self.last_motion = None
        self.last_health = None
        self.last_hit = None

    def angles(self, x, y):
        yaw = math.degrees(math.atan((x - self.width / 2) / self.fx))
        pitch = math.degrees(math.atan((self.height / 2 - y) / self.fx))
        return yaw, pitch

    def depth_in_box(self, depth, box):
        """Median relative depth in the box's central area, in depth-map coordinates."""
        d = np.asarray(depth)
        sy, sx = d.shape[0] / self.height, d.shape[1] / self.width
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx1, cx2 = int((x1 + 0.3 * bw) * sx), int((x2 - 0.3 * bw) * sx) + 1
        cy1, cy2 = int((y1 + 0.2 * bh) * sy), int((y1 + 0.6 * bh) * sy) + 1
        patch = d[max(0, cy1):cy2, max(0, cx1):cx2]
        return float(np.median(patch)) if patch.size else None

    def read_ammo(self, reading, t):
        """A count only counts once OCR has read the same value ammo_confirm times in a row:
        one stray digit in the HUD corner must not start a reload loop."""
        if reading[0] is None:
            self.ammo_candidate = (None, 0)
            return
        value, times = self.ammo_candidate
        times = times + 1 if value == reading else 1
        self.ammo_candidate = (reading, times)
        if times >= self.cfg.ammo_confirm:
            self.ammo, self.ammo_t = reading, t

    def to_screen(self, yaw, pitch):
        """Inverse of angles(): degrees from the crosshair -> pixel."""
        return (self.width / 2 + self.fx * math.tan(math.radians(yaw)),
                self.height / 2 - self.fx * math.tan(math.radians(pitch)))

    def move_remembered(self, tr, dyaw, dpitch):
        """Shift a remembered target's angle from the crosshair by (dyaw, dpitch) and re-project it."""
        tr.yaw_err = max(-85.0, min(85.0, tr.yaw_err + dyaw))
        tr.pitch_err = max(-85.0, min(85.0, tr.pitch_err + dpitch))
        nx, ny = self.to_screen(tr.yaw_err, tr.pitch_err)
        sx, sy = nx - tr.aim[0], ny - tr.aim[1]
        tr.aim = (nx, ny)
        x1, y1, x2, y2 = tr.box
        tr.box = (x1 + sx, y1 + sy, x2 + sx, y2 + sy)

    def rate(self, prev, yaw, pitch, camera, t):
        """World angular velocity of a visible target, from its previous frame (deg/s), smoothed."""
        cfg = self.cfg
        dt = 1.0 / self.fps
        if prev is None or not prev.visible or t - prev.last_seen > 1.5 * dt:
            # new, or back after a gap: no fresh sample, keep what we knew
            return (prev.yaw_rate, prev.pitch_rate, 0) if prev else (0.0, 0.0, 0)
        cam_yaw, cam_pitch = camera if camera is not None else (0.0, 0.0)
        # on screen it moved by (yaw - prev); the camera's own turn hid cam_yaw of its movement
        raw_yaw = (yaw - prev.yaw_err + cam_yaw) / dt
        raw_pitch = (pitch - prev.pitch_err + cam_pitch) / dt
        m = cfg.max_rate_dps
        raw_yaw, raw_pitch = max(-m, min(m, raw_yaw)), max(-m, min(m, raw_pitch))
        if prev.rate_samples == 0:
            return raw_yaw, raw_pitch, 1
        a = cfg.rate_alpha
        return ((1 - a) * prev.yaw_rate + a * raw_yaw, (1 - a) * prev.pitch_rate + a * raw_pitch,
                prev.rate_samples + 1)

    def calibrate_depth(self, kept):
        """Update k (depth_rel = k / dist) from detections that look like whole standing people."""
        ks = []
        for obj, dist_box, depth_rel in kept:
            x1, y1, x2, y2 = obj["box"]
            if depth_rel and depth_rel > 0 and (y2 - y1) >= self.cfg.standing_aspect * (x2 - x1):
                ks.append(depth_rel * dist_box)
        if ks:
            k = float(np.median(ks))
            a = self.cfg.depth_scale_alpha
            self.depth_scale = k if self.depth_scale is None else (1 - a) * self.depth_scale + a * k

    def update(self, index, results):
        cfg = self.cfg
        t = index / self.fps
        depth = results.get("depth")
        seen = set()
        motion = results.get("motion")
        camera = None
        if motion is not None and motion is not self.last_motion:  # a new estimate this frame
            self.last_motion = motion
            if motion.get("valid"):
                camera = (motion["yaw"], motion["pitch"])

        kept = []
        for obj in results.get("objects") or []:
            x1, y1, x2, y2 = obj["box"]
            bh = y2 - y1
            if obj.get("conf", 1.0) < cfg.min_conf or bh < cfg.min_box_h_px or bh > cfg.max_box_h_frac * self.height:
                continue
            dist_box = cfg.person_height_m * self.fx / bh
            depth_rel = self.depth_in_box(depth, obj["box"]) if depth is not None else None
            kept.append((obj, dist_box, depth_rel))

        if cfg.use_depth_distance:
            self.calibrate_depth(kept)

        for obj, dist_box, depth_rel in kept:
            x1, y1, x2, y2 = obj["box"]
            bh = y2 - y1
            tid = obj["id"]
            aim = ((x1 + x2) / 2, y1 + cfg.aim_from_top * bh)
            yaw, pitch = self.angles(*aim)
            width_deg = abs(self.angles(x2, aim[1])[0] - self.angles(x1, aim[1])[0])
            dist_depth = None
            if cfg.use_depth_distance and self.depth_scale and depth_rel and depth_rel > 0:
                dist_depth = self.depth_scale / depth_rel
            distance = dist_depth if dist_depth is not None else dist_box
            prev = self.tracks.get(tid)
            yaw_rate, pitch_rate, samples = self.rate(prev, yaw, pitch, camera, t)
            target = Target(
                yaw_rate=yaw_rate, pitch_rate=pitch_rate, rate_samples=samples,
                id=tid, box=tuple(obj["box"]), aim=aim, yaw_err=yaw, pitch_err=pitch, width_deg=width_deg,
                distance_m=distance, range=range_class(distance, prev.range if prev else None, cfg),
                conf=obj.get("conf", 1.0), first_seen=prev.first_seen if prev else t, last_seen=t,
                frames_seen=(prev.frames_seen + 1) if prev else 1, visible=True,
                foe=obj.get("foe", prev.foe if prev else None),
                depth_rel=depth_rel, dist_box_m=dist_box, dist_depth_m=dist_depth,
            )
            self.tracks[tid] = target
            seen.add(tid)

        for tid in list(self.tracks):
            if tid in seen:
                continue
            tr = self.tracks[tid]
            if t - tr.last_seen > cfg.memory_s or tr.frames_seen < cfg.min_frames_seen:
                del self.tracks[tid]  # forgotten, or never confirmed
            else:
                tr.visible = False
                # the view turned right -> it's further left; and it keeps moving as it was, for a while
                dyaw, dpitch = (-camera[0], -camera[1]) if camera is not None else (0.0, 0.0)
                if tr.rate_samples >= cfg.min_rate_samples and t - tr.last_seen <= cfg.predict_s:
                    dyaw += tr.yaw_rate / self.fps
                    dpitch += tr.pitch_rate / self.fps
                if dyaw or dpitch:
                    self.move_remembered(tr, dyaw, dpitch)

        text = results.get("text")
        if text is not None and text is not self.last_text:  # text runs every Nth frame: parse new results only
            self.last_text = text
            self.read_ammo(parse_ammo(text, self.width, self.height, cfg.ammo_region), t)
        if self.ammo_t is not None and t - self.ammo_t > cfg.ammo_expiry_s:
            self.ammo, self.ammo_t = (None, None), None  # stale: better unknown than wrong

        health = results.get("health")
        if health is not None and health is not self.last_health:  # health runs every 2nd frame
            self.last_health = health
            if health.get("hit"):
                self.last_hit = t
        me = SelfState(last_hit=self.last_hit,ammo=self.ammo[0], reserve=self.ammo[1],
                       health=health["health"] if health else None,
                       blocked=blocked_ahead(depth) if depth is not None else {})
        confirmed = [tr for tr in self.tracks.values() if tr.frames_seen >= cfg.min_frames_seen]
        covers = []
        if depth is not None:
            covers = find_cover(depth, self.width, self.height, self.fx, self.depth_scale,
                                exclude_boxes=[o["box"] for o, _, _ in kept], max_m=cfg.cover_max_m,
                                min_width_deg=cfg.cover_min_width_deg)
        return WorldState(index=index, t=t, width=self.width, height=self.height, targets=confirmed, me=me,
                          camera=camera, has_motion=motion is not None, covers=covers, fx=self.fx)

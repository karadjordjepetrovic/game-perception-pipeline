"""Hierarchical state machine for first-person-shooter input.

Top level, highest priority first:

    RECOVER   health low: back off, no shooting            (needs a health stage; idle until then)
    RELOAD    magazine empty, or low with no close threat  (runs for reload_s once started)
    ENGAGE    a confirmed foe is visible
      ├─ APPROACH  target far: move toward it, keep the aim roughly on it
      ├─ ACQUIRE   near/mid, aim error above tolerance: turn onto it
      └─ FIRE      near/mid, on target: bursts, re-aiming between them
    SEARCH    a foe was lost < memory_s ago: turn toward where it was last seen
    EXPLORE   nothing known: walk forward, turn away from walls

Against flicker from noisy per-frame perception:
- a state with higher priority can take over at once; switching to a lower one needs the
  current state to have lasted min_state_frames (dwell time)
- FIRE is entered at the aim tolerance but only left at exit_factor x tolerance (hysteresis)
- target selection gives the current target a stickiness bonus
- range classes have their own hysteresis (gpp.world.range_class)

Target choice (target_policy):
- "closest" (default): the nearest foe by gpp.world's fused distance; the agent only switches
  when another foe is at least switch_margin (20%) closer than the current one
- "score": weighted mix of closeness, distance from the crosshair and stickiness

Aim (aim_mode):
- "smooth" (default): turn gradually, fire once on target; for a closed loop (a real game)
- "snap": turn the whole error in one tick and fire on the same tick, "shoot at (x, y)";
  errors above max_snap_deg are turned smoothly first. Fits recorded footage, where the
  crosshair never follows the agent and smooth aim would almost never reach FIRE
Every ENGAGE action carries the target's screen point and angle offset (aim_px, aim_deg).

Short detection gaps (< lost_grace_s): the agent keeps aiming at the target's last position
and keeps the current burst going, instead of freezing.

Look control is a proportional controller written as a time constant: each tick turns the
fraction 1 - exp(-dt / look_tau_s) of the remaining error, capped at max_turn_dps. That is
the same behaviour at 30 and 60 fps.

On recorded footage the loop is open: the agent's turns don't change the next frame, so each
decision is "what to do given this frame". SEARCH turns toward the lost target's current
position: with the motion stage that position follows the camera (it works the same on
footage and in a live game); without it, SEARCH counts its own turns instead.
"""
import math
from dataclasses import dataclass
from typing import Optional

from ..world import FAR, NEAR, WorldState
from .actions import Action

RECOVER, RELOAD, ENGAGE, SEARCH, EXPLORE = "RECOVER", "RELOAD", "ENGAGE", "SEARCH", "EXPLORE"
APPROACH, ACQUIRE, FIRE = "APPROACH", "ACQUIRE", "FIRE"
PRIORITY = {RECOVER: 5, RELOAD: 4, ENGAGE: 3, SEARCH: 2, EXPLORE: 1}


@dataclass
class AgentConfig:
    min_state_frames: int = 6        # dwell before switching to a lower-priority state
    look_tau_s: float = 0.08         # look controller time constant
    max_turn_dps: float = 360.0      # turn rate cap, degrees per second
    aim_tol_frac: float = 0.35       # on target when error < this x target's angular width
    min_aim_tol_deg: float = 0.5
    max_aim_tol_deg: float = 4.0
    exit_factor: float = 1.5         # leave FIRE above exit_factor x tolerance
    lost_grace_s: float = 0.3        # target gone this briefly: stay in ENGAGE (detection dropouts)
    burst_s: float = 0.15            # fire this long ...
    burst_pause_s: float = 0.10      # ... then pause to re-check aim
    fire_at_far: bool = False
    reload_s: float = 2.5
    low_ammo: int = 5
    ammo_ignore_s: float = 1.0       # after a reload, ignore ammo readings this long (HUD/OCR lag)
    blocked_threshold: float = 0.85  # EXPLORE turns when the path ahead is closer than this (0..1)
    low_health: float = 0.3
    explore_turn_dps: float = 90.0
    search_overshoot_deg: float = 10.0
    target_policy: str = "closest"   # "closest" or "score"
    switch_margin: float = 0.2       # closest: switch only to a foe this much closer than the current one
    aim_mode: str = "smooth"         # "smooth" or "snap"
    max_snap_deg: float = 45.0       # snap: larger errors are turned smoothly first
    w_close: float = 1.0             # "score" policy weights
    w_centre: float = 1.0
    w_sticky: float = 0.5


@dataclass
class Decision:
    state: str                       # "ENGAGE.FIRE"
    action: Action
    target: Optional[int]
    reason: str

    def to_dict(self):
        return {"state": self.state, "action": self.action.to_dict(), "target": self.target, "reason": self.reason}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FpsStateMachine:
    def __init__(self, fps, hfov_deg=80.0, cfg=None, assume_foe=True):
        self.cfg = cfg or AgentConfig()
        self.dt = 1.0 / fps
        self.hfov = hfov_deg
        self.assume_foe = assume_foe
        self.mode = EXPLORE
        self.sub = None
        self.frames_in_mode = 0
        self.target_id = None
        self.reload_until = -1.0
        self.ammo_ignore_until = -1.0
        self.fire_phase_t = 0.0
        self.search_yaw_left = 0.0
        self.search_target = None

    # --- helpers ---------------------------------------------------------------------------
    def look(self, yaw_err, pitch_err=0.0):
        """Degrees to turn this tick toward an error (P-controller with a rate cap)."""
        frac = 1.0 - math.exp(-self.dt / self.cfg.look_tau_s)
        cap = self.cfg.max_turn_dps * self.dt
        return _clamp(yaw_err * frac, -cap, cap), _clamp(pitch_err * frac, -cap, cap)

    def aim_tolerance(self, target):
        c = self.cfg
        return _clamp(c.aim_tol_frac * target.width_deg, c.min_aim_tol_deg, c.max_aim_tol_deg)

    def select_target(self, foes):
        c = self.cfg
        if c.target_policy == "closest":
            best = min(foes, key=lambda t: t.distance_m)
            current = next((t for t in foes if t.id == self.target_id), None)
            if current is not None and best.distance_m >= current.distance_m * (1.0 - c.switch_margin):
                return current  # not clearly closer: keep the current target
            return best

        def score(t):
            close = 1.0 / (1.0 + t.distance_m / 10.0)
            centre = 1.0 - min(1.0, math.hypot(t.yaw_err, t.pitch_err) / (self.hfov / 2))
            return c.w_close * close + c.w_centre * centre + (c.w_sticky if t.id == self.target_id else 0.0)

        return max(foes, key=score)

    def burst(self):
        """Fire / pause cycle while on target."""
        c = self.cfg
        firing = (self.fire_phase_t % (c.burst_s + c.burst_pause_s)) < c.burst_s
        self.fire_phase_t += self.dt
        return firing

    # --- top level -------------------------------------------------------------------------
    def desired_mode(self, w: WorldState):
        c = self.cfg
        foes = w.visible_foes(self.assume_foe)
        if w.me.health is not None and w.me.health <= c.low_health:
            return RECOVER, f"health {w.me.health:.0%} <= {c.low_health:.0%}"
        if w.t < self.reload_until:
            return RELOAD, "reloading"
        ammo = w.me.ammo
        if ammo is not None and w.t >= self.ammo_ignore_until:
            close = [f for f in foes if f.range != FAR]
            if ammo == 0:
                return RELOAD, "magazine empty"
            if ammo <= c.low_ammo and not close:
                return RELOAD, f"ammo {ammo} <= {c.low_ammo}, no close threat"
        if foes:
            return ENGAGE, f"{len(foes)} foe(s) visible"
        lost = w.lost_foes(self.assume_foe)
        if self.mode == ENGAGE and any(t.id == self.target_id and w.t - t.last_seen < c.lost_grace_s for t in lost):
            return ENGAGE, "target briefly out of view"
        if lost:
            return SEARCH, "foe lost"
        return EXPLORE, "nothing in view"

    def step(self, w: WorldState) -> Decision:
        desired, why = self.desired_mode(w)
        if desired != self.mode:
            if PRIORITY[desired] > PRIORITY[self.mode] or self.frames_in_mode >= self.cfg.min_state_frames:
                self.enter(desired, w)
        self.frames_in_mode += 1
        handler = {RECOVER: self.recover, RELOAD: self.reload, ENGAGE: self.engage,
                   SEARCH: self.search, EXPLORE: self.explore}[self.mode]
        action, target, reason = handler(w)
        if self.mode != desired:
            reason += f" (holding {self.mode}, wants {desired}: {why})"
        state = f"{self.mode}.{self.sub}" if self.sub else self.mode
        return Decision(state, action, target, reason)

    def enter(self, mode, w):
        if self.mode == ENGAGE and mode == SEARCH:
            lost = [t for t in w.lost_foes(self.assume_foe) if t.id == self.target_id] or w.lost_foes(self.assume_foe)
            last = max(lost, key=lambda t: t.last_seen)
            self.search_target = last.id
            overshoot = math.copysign(self.cfg.search_overshoot_deg, last.yaw_err)
            self.search_yaw_left = last.yaw_err + overshoot
        elif mode == SEARCH:
            last = max(w.lost_foes(self.assume_foe), key=lambda t: t.last_seen)
            self.search_target, self.search_yaw_left = last.id, last.yaw_err
        if mode == RELOAD and w.t >= self.reload_until:
            self.reload_until = w.t + self.cfg.reload_s
            # the HUD keeps showing the old count until the next OCR result: don't re-trigger on it
            self.ammo_ignore_until = self.reload_until + self.cfg.ammo_ignore_s
        self.mode, self.sub, self.frames_in_mode = mode, None, 0
        self.fire_phase_t = 0.0

    # --- states ----------------------------------------------------------------------------
    def recover(self, w):
        b = w.me.blocked
        strafe = 0.0
        if b:
            strafe = -1.0 if b.get("left", 1) < b.get("right", 1) else 1.0
        return Action(forward=-1.0, strafe=strafe), None, "backing off to recover"

    def reload(self, w):
        starting = self.frames_in_mode == 1
        near = [f for f in w.visible_foes(self.assume_foe) if f.range == NEAR]
        action = Action(reload=starting, forward=-1.0 if near else 0.0)
        left = max(0.0, self.reload_until - w.t)
        reason = "start reload" if starting else f"reloading, {left:.1f}s left"
        if near:
            reason += f", backing away from #{near[0].id}"
        return action, None, reason

    def engage(self, w):
        c = self.cfg
        foes = w.visible_foes(self.assume_foe)
        if not foes:
            return self.coast(w)
        target = self.select_target(foes)
        if target.id != self.target_id and self.target_id is not None:
            self.fire_phase_t = 0.0
        self.target_id = target.id
        err = math.hypot(target.yaw_err, target.pitch_err)
        tol = self.aim_tolerance(target)
        side = "right" if target.yaw_err > 0 else "left"
        vert = "up" if target.pitch_err > 0 else "down"
        desc = f"#{target.id} {target.range} ~{target.distance_m:.0f}m"
        where = f"({target.aim[0]:.0f}, {target.aim[1]:.0f}), {abs(target.yaw_err):.1f}° {side} {abs(target.pitch_err):.1f}° {vert}"
        aim = dict(aim_px=target.aim, aim_deg=(target.yaw_err, target.pitch_err))
        yaw, pitch = self.look(target.yaw_err, target.pitch_err)

        if target.range == FAR and not c.fire_at_far:
            self.sub = APPROACH
            return Action(forward=1.0, yaw=yaw, pitch=pitch, **aim), target.id, f"{desc} at {where}: closing distance"

        if c.aim_mode == "snap" and err <= c.max_snap_deg:
            if self.sub != FIRE:
                self.sub, self.fire_phase_t = FIRE, 0.0
            firing = self.burst()
            return (Action(yaw=target.yaw_err, pitch=target.pitch_err, ads=True, fire=firing, **aim), target.id,
                    f"{'shoot' if firing else 'burst pause,'} {desc} at {where}")

        on_target = err <= tol * (c.exit_factor if self.sub == FIRE else 1.0)
        if not on_target:
            self.sub = ACQUIRE
            return (Action(yaw=yaw, pitch=pitch, ads=True, **aim), target.id,
                    f"{desc} at {where}: aligning (err {err:.1f}° > {tol:.1f}°)")

        if self.sub != FIRE:
            self.sub, self.fire_phase_t = FIRE, 0.0
        firing = self.burst()
        return (Action(yaw=yaw, pitch=pitch, ads=True, fire=firing, **aim), target.id,
                f"{desc} at {where}: on target (err {err:.1f}° <= {tol:.1f}°), {'firing' if firing else 'burst pause'}")

    def coast(self, w):
        """Target briefly undetected: keep aiming where it was last seen, keep the burst going."""
        last = next((t for t in w.lost_foes(self.assume_foe) if t.id == self.target_id), None)
        if last is None:  # held in ENGAGE only by the dwell time
            self.sub = None
            return Action(ads=True), self.target_id, "target out of view"
        aim = dict(aim_px=last.aim, aim_deg=(last.yaw_err, last.pitch_err))
        if self.cfg.aim_mode == "snap":
            yaw, pitch = last.yaw_err, last.pitch_err
        else:
            yaw, pitch = self.look(last.yaw_err, last.pitch_err)
        firing = self.burst() if self.sub == FIRE else False
        gone = w.t - last.last_seen
        return (Action(yaw=yaw, pitch=pitch, ads=True, fire=firing, **aim), last.id,
                f"#{last.id} out of view {gone * 1000:.0f} ms, holding aim on last position")

    def search(self, w):
        self.target_id = None
        cap = self.cfg.max_turn_dps * self.dt * 0.5  # search turns at half speed
        if w.has_motion:
            lost = w.lost_foes(self.assume_foe)
            last = next((t for t in lost if t.id == self.search_target), None) or max(lost, key=lambda t: t.last_seen)
            self.search_target = last.id
            if abs(last.yaw_err) > 2.0 or abs(last.pitch_err) > 2.0:
                yaw, pitch = self.look(last.yaw_err, last.pitch_err)
                yaw, pitch = _clamp(yaw, -cap, cap), _clamp(pitch, -cap, cap)
                side = "right" if last.yaw_err > 0 else "left"
                return (Action(yaw=yaw, pitch=pitch, aim_px=last.aim, aim_deg=(last.yaw_err, last.pitch_err)), last.id,
                        f"turning toward #{last.id}, now {abs(last.yaw_err):.1f}° {side} (followed with camera)")
            return (Action(aim_px=last.aim, aim_deg=(last.yaw_err, last.pitch_err)), last.id,
                    f"watching where #{last.id} was last seen")
        turn = _clamp(self.search_yaw_left, -cap, cap)
        self.search_yaw_left -= turn
        if abs(turn) > 1e-3:
            return Action(yaw=turn), self.search_target, f"turning toward last seen #{self.search_target}"
        return Action(), self.search_target, f"watching where #{self.search_target} was last seen"

    def explore(self, w):
        self.target_id = None
        b = w.me.blocked
        if b and b.get("centre", 0.0) > self.cfg.blocked_threshold:
            direction = -1.0 if b.get("left", 1.0) < b.get("right", 1.0) else 1.0
            turn = direction * self.cfg.explore_turn_dps * self.dt
            side = "left" if direction < 0 else "right"
            return Action(yaw=turn), None, f"path blocked ({b['centre']:.2f}), turning {side}"
        return Action(forward=1.0), None, "path clear, moving forward" if b else "no depth, moving forward"

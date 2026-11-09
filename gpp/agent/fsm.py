"""Hierarchical state machine for first-person-shooter input.

Top level, highest priority first:

    RECOVER   hit (or health low): no shooting; walk to the safest cover in view while
              aiming at the threat, then crouch (gpp/cover.py); back off if there's none
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

Lead (lead_s): the agent aims where a moving target will be lead_s from now, using its
world angular velocity from gpp.world (minus nothing for the camera: the aim is relative to
the crosshair now). Off for targets whose velocity isn't trusted yet.

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

from ..cover import choose_hide
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
    recover_on_hit: bool = True      # RECOVER as soon as the player is hit ...
    recover_s: float = 2.0           # ... for this long after the last hit
    low_health: float = 0.3          # also RECOVER below this health ...
    recover_until: float = 0.6       # ... until it is back above this
    use_cover: bool = True           # RECOVER walks to cover (else just backs off)
    cover_offset_m: float = 1.0      # hide this far behind the cover object
    cover_arrive_m: float = 0.75     # this close to the hiding spot: crouch and hold
    explore_turn_dps: float = 90.0
    search_overshoot_deg: float = 10.0
    target_policy: str = "closest"   # "closest" or "score"
    switch_margin: float = 0.2       # closest: switch only to a foe this much closer than the current one
    aim_mode: str = "smooth"         # "smooth" or "snap"
    max_snap_deg: float = 45.0       # snap: larger errors are turned smoothly first
    lead_s: float = 0.05             # aim this far ahead of a moving target (0 = no lead)
    min_rate_samples: int = 3        # velocity needs this many frames before it is used for lead
    min_lead_dps: float = 10.0       # slower readings are mostly box jitter: no lead
    w_close: float = 1.0             # "score" policy weights
    w_centre: float = 1.0
    w_sticky: float = 0.5


@dataclass
class Decision:
    state: str                       # "ENGAGE.FIRE"
    action: Action
    target: Optional[int]
    reason: str
    cover: Optional[object] = None   # gpp.cover.HideChoice while taking cover

    def to_dict(self):
        d = {"state": self.state, "action": self.action.to_dict(), "target": self.target, "reason": self.reason}
        if self.cover is not None:
            d["cover"] = {**self.cover.cover.to_dict(), "covered": self.cover.covered, "threats": self.cover.threats}
        return d


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

    def to_screen(self, w, yaw, pitch):
        """Degrees from the crosshair -> pixel (same pinhole camera as gpp.world)."""
        fx = (w.width / 2) / math.tan(math.radians(self.hfov) / 2)
        return (w.width / 2 + fx * math.tan(math.radians(yaw)), w.height / 2 - fx * math.tan(math.radians(pitch)))

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
        hp = w.me.health
        if c.recover_on_hit and w.me.last_hit is not None and w.t - w.me.last_hit < c.recover_s:
            ago = w.t - w.me.last_hit
            return RECOVER, f"hit {ago * 1000:.0f} ms ago" + (f", health {hp:.0%}" if hp is not None else "")
        if hp is not None and (hp <= c.low_health or (self.mode == RECOVER and hp < c.recover_until)):
            limit = c.low_health if self.mode != RECOVER else c.recover_until
            return RECOVER, f"health {hp:.0%} {'<=' if self.mode != RECOVER else '<'} {limit:.0%}"
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
        self.cover = None
        action, target, reason = handler(w)
        if self.mode != desired:
            reason += f" (holding {self.mode}, wants {desired}: {why})"
        state = f"{self.mode}.{self.sub}" if self.sub else self.mode
        return Decision(state, action, target, reason, cover=self.cover)

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
        c = self.cfg
        if w.me.last_hit is not None and w.t - w.me.last_hit < c.recover_s:
            why = f"hit {(w.t - w.me.last_hit) * 1000:.0f} ms ago"
        else:
            why = f"health {w.me.health:.0%}" if w.me.health is not None else "low health"
        threats = w.visible_foes(self.assume_foe) + w.lost_foes(self.assume_foe)
        nearest = min(threats, key=lambda t: t.distance_m) if threats else None
        look = self.look(nearest.yaw_err, nearest.pitch_err) if nearest else (0.0, 0.0)
        choice = choose_hide(w.covers, [(t.yaw_err, t.distance_m) for t in threats],
                             offset_m=c.cover_offset_m) if c.use_cover else None
        if choice is not None:
            self.cover = choice
            cv = choice.cover
            side = "right" if cv.yaw > 0 else "left"
            obj = f"object {abs(cv.yaw):.0f}° {side} ~{cv.dist_m:.1f} m" + ("" if cv.metric else " (assumed)")
            blocks = f"blocks {choice.covered}/{choice.threats} threats" if choice.threats else "no threat seen"
            hx, hz = choice.hide
            if choice.dist_m <= c.cover_arrive_m:
                return (Action(yaw=look[0], pitch=look[1], crouch=True, move_to=choice.hide),
                        nearest.id if nearest else None, f"{why}: in cover behind {obj}, crouched ({blocks})")
            return (Action(forward=hz / choice.dist_m, strafe=hx / choice.dist_m, yaw=look[0], pitch=look[1],
                           move_to=choice.hide), nearest.id if nearest else None,
                    f"{why}: taking cover behind {obj}, {choice.dist_m:.1f} m to go ({blocks})")
        b = w.me.blocked
        strafe = 0.0
        if b:
            strafe = -1.0 if b.get("left", 1) < b.get("right", 1) else 1.0
        if w.me.last_hit is not None and w.t - w.me.last_hit < self.cfg.recover_s:
            hp = f"hit {(w.t - w.me.last_hit) * 1000:.0f} ms ago"
        else:
            hp = f"health {w.me.health:.0%}" if w.me.health is not None else "low health"
        side = "" if not strafe else (" and strafing left" if strafe < 0 else " and strafing right")
        return (Action(forward=-1.0, strafe=strafe, yaw=look[0], pitch=look[1]), nearest.id if nearest else None,
                f"{hp}: no cover in view, backing off{side}")

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
        lead_yaw = lead_pitch = 0.0
        if (c.lead_s > 0 and target.rate_samples >= c.min_rate_samples
                and math.hypot(target.yaw_rate, target.pitch_rate) >= c.min_lead_dps):
            lead_yaw, lead_pitch = target.yaw_rate * c.lead_s, target.pitch_rate * c.lead_s
        aim_yaw, aim_pitch = target.yaw_err + lead_yaw, target.pitch_err + lead_pitch
        err = math.hypot(aim_yaw, aim_pitch)
        tol = self.aim_tolerance(target)
        desc = f"#{target.id} {target.range} ~{target.distance_m:.0f}m"
        side = "right" if aim_yaw > 0 else "left"
        vert = "up" if aim_pitch > 0 else "down"
        px = self.to_screen(w, aim_yaw, aim_pitch)
        where = f"({px[0]:.0f}, {px[1]:.0f}), {abs(aim_yaw):.1f}° {side} {abs(aim_pitch):.1f}° {vert}"
        if abs(lead_yaw) >= 0.1 or abs(lead_pitch) >= 0.1:
            where += f", lead {lead_yaw:+.1f}°"
        aim = dict(aim_px=px, aim_deg=(aim_yaw, aim_pitch))
        yaw, pitch = self.look(aim_yaw, aim_pitch)

        if target.range == FAR and not c.fire_at_far:
            self.sub = APPROACH
            return Action(forward=1.0, yaw=yaw, pitch=pitch, **aim), target.id, f"{desc} at {where}: closing distance"

        if c.aim_mode == "snap" and err <= c.max_snap_deg:
            if self.sub != FIRE:
                self.sub, self.fire_phase_t = FIRE, 0.0
            firing = self.burst()
            return (Action(yaw=aim_yaw, pitch=aim_pitch, ads=True, fire=firing, **aim), target.id,
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

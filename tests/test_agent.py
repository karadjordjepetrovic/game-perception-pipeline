"""Decision agent tests: synthetic stage results in, decisions out. No GPU, models or footage."""
import json
import math

import cv2
import numpy as np
import pytest

from gpp import pipeline, report
from gpp.agent import AgentConfig, DecisionAgent, FpsStateMachine, WorldConfig
from gpp.overlay import draw
from gpp.stage import Stage
from gpp.video import VideoInfo, VideoReader, probe
from gpp.world import FAR, MID, NEAR, WorldBuilder, parse_ammo, range_class

W, H, FPS = 1920, 1080, 60
INFO = VideoInfo("synthetic", W, H, FPS, 0)
FX = (W / 2) / math.tan(math.radians(80) / 2)


def person(tid, dist_m, yaw_deg=0.0, pitch_deg=0.0, conf=0.9):
    """A detection whose aim point is at (yaw, pitch) from the crosshair and whose height means dist_m."""
    bh = 1.8 * FX / dist_m
    bw = 0.4 * bh
    ax = W / 2 + FX * math.tan(math.radians(yaw_deg))
    ay = H / 2 - FX * math.tan(math.radians(pitch_deg))
    y1 = ay - 0.2 * bh
    return {"box": (ax - bw / 2, y1, ax + bw / 2, y1 + bh), "id": tid, "conf": conf}


def hud(n):
    """A fresh OCR result with the ammo count (a new list, as the text stage gives on each OCR frame)."""
    return [{"box": (1700, 950, 1760, 990), "text": str(n), "label": "hud"}]


def depth_for(objs, k=100.0, background=0.5, size=(270, 480)):
    """A depth map (relative inverse depth = k / dist) with each detection's area at its true distance.

    objs: list of (detection, true_distance_m). Map is smaller than the frame, like the real one."""
    d = np.full(size, background, dtype=np.float32)
    sy, sx = size[0] / H, size[1] / W
    for obj, dist in objs:
        x1, y1, x2, y2 = obj["box"]
        d[int(y1 * sy):int(y2 * sy) + 1, int(x1 * sx):int(x2 * sx) + 1] = k / dist
    return d


def crouched(tid, dist_m, yaw_deg=0.0):
    """Half-height box (crouching or behind cover): box height says twice as far as it is."""
    p = person(tid, dist_m, yaw_deg)
    x1, y1, x2, y2 = p["box"]
    p["box"] = (x1, y1, x2, y1 + (y2 - y1) / 2)
    return p


def make_agent(**cfg):
    agent = DecisionAgent(agent_cfg=AgentConfig(**cfg))
    agent.setup(INFO)
    return agent


def feed(agent, frames, start=0):
    """frames: list of results dicts. Returns the decisions."""
    return [agent.process(start + i, r)["decision"] for i, r in enumerate(frames)]


# --- world --------------------------------------------------------------------------------
def test_angles_and_distance_from_box():
    b = WorldBuilder(W, H, FPS)
    for i in range(3):
        world = b.update(i, {"objects": [person(1, 12.0, yaw_deg=10.0, pitch_deg=-3.0)]})
    (t,) = world.targets
    assert t.yaw_err == pytest.approx(10.0, abs=0.05)
    assert t.pitch_err == pytest.approx(-3.0, abs=0.05)
    assert t.distance_m == pytest.approx(12.0, rel=0.01)
    assert t.range == MID


def test_range_hysteresis():
    cfg = WorldConfig(near_m=10, far_m=30, range_margin=0.15)
    assert range_class(10.5, None, cfg) == MID
    assert range_class(10.5, NEAR, cfg) == NEAR       # within the margin: stays near
    assert range_class(12.0, NEAR, cfg) == MID
    assert range_class(27.0, FAR, cfg) == FAR
    assert range_class(24.0, FAR, cfg) == MID


def test_unconfirmed_and_oversized_boxes_ignored():
    b = WorldBuilder(W, H, FPS)
    world = b.update(0, {"objects": [person(1, 12.0)]})
    assert world.targets == []                        # seen once: not confirmed yet
    arms = {"box": (600, 50, 1500, 1080), "id": 2, "conf": 0.9}
    for i in range(5):
        world = b.update(i, {"objects": [arms]})
    assert world.targets == []                        # too tall: the player's own viewmodel


def test_parse_ammo():
    hud = lambda text, x, y, h=30: {"box": (x, y, x + 60, y + h), "text": text, "label": "hud"}
    texts = [hud("24", 1700, 950, 40), hud("120", 1780, 960, 20), hud("12", 100, 100),
             {"box": (1700, 900, 1760, 930), "text": "7", "label": "world"}]
    assert parse_ammo(texts, W, H, (0.6, 0.6, 1, 1)) == (24, None)
    assert parse_ammo([hud("30 / 90", 1700, 950)], W, H, (0.6, 0.6, 1, 1)) == (30, 90)
    assert parse_ammo([hud("AMMO", 1700, 950)], W, H, (0.6, 0.6, 1, 1)) == (None, None)


# --- state machine ------------------------------------------------------------------------
def test_explore_when_nothing_seen_and_single_frame_detection_ignored():
    agent = make_agent()
    d = feed(agent, [{"objects": []}, {"objects": [person(1, 8.0)]}, {"objects": []}])
    assert [x.state for x in d] == ["EXPLORE"] * 3
    assert d[0].action.forward == 1.0


def test_far_target_approach():
    agent = make_agent()
    d = feed(agent, [{"objects": [person(1, 60.0, yaw_deg=5.0)]}] * 4)
    assert d[-1].state == "ENGAGE.APPROACH"
    assert d[-1].action.forward == 1.0 and d[-1].action.yaw > 0 and not d[-1].action.fire


def test_acquire_then_fire():
    agent = make_agent()
    off = feed(agent, [{"objects": [person(1, 6.0, yaw_deg=10.0)]}] * 4)
    assert off[-1].state == "ENGAGE.ACQUIRE"
    a = off[-1].action
    assert 0 < a.yaw < 10.0 and a.ads and not a.fire
    on = feed(agent, [{"objects": [person(1, 6.0, yaw_deg=0.3)]}], start=4)
    assert on[0].state == "ENGAGE.FIRE" and on[0].action.fire and on[0].target == 1


def test_fire_hysteresis_and_bursts():
    agent = make_agent(lead_s=0.0)  # the test's sudden jump would read as movement
    feed(agent, [{"objects": [person(1, 6.0)]}] * 4)
    tol = agent.fsm.aim_tolerance(agent.builder.tracks[1])
    # just above the entry tolerance but below exit: stays in FIRE
    d = feed(agent, [{"objects": [person(1, 6.0, yaw_deg=tol * 1.2)]}] * 60, start=4)
    assert all(x.state == "ENGAGE.FIRE" for x in d)
    fired = sum(x.action.fire for x in d) / len(d)
    assert fired == pytest.approx(0.15 / 0.25, abs=0.05)  # burst 0.15 s, pause 0.10 s
    out = feed(agent, [{"objects": [person(1, 6.0, yaw_deg=tol * 2.0)]}], start=64)
    assert out[0].state == "ENGAGE.ACQUIRE"


def test_prefers_close_centred_target_and_sticks():
    agent = make_agent()
    d = feed(agent, [{"objects": [person(1, 40.0, yaw_deg=1.0), person(2, 6.0, yaw_deg=3.0)]}] * 4)
    assert d[-1].target == 2
    # a slightly better-placed target appears: stickiness keeps the current one
    d = feed(agent, [{"objects": [person(2, 6.0, yaw_deg=3.0), person(3, 6.0, yaw_deg=2.5)]}] * 4, start=4)
    assert d[-1].target == 2


def test_lost_target_search_then_explore():
    agent = make_agent()
    feed(agent, [{"objects": [person(1, 8.0, yaw_deg=-20.0)]}] * 10)
    lost = feed(agent, [{"objects": []}] * (3 * FPS), start=10)
    states = [x.state for x in lost]
    first = states.index("SEARCH")
    assert first == pytest.approx(0.3 * FPS, abs=2)    # grace period for detection dropouts first
    assert lost[first].action.yaw < 0                  # turns toward where it was (left)
    turned = sum(x.action.yaw for x in lost if x.state == "SEARCH")
    assert turned == pytest.approx(-30.0, abs=0.5)     # last seen at -20°, plus 10° overshoot
    assert states[-1] == "EXPLORE"
    assert states.index("EXPLORE") == pytest.approx(2.0 * FPS, abs=3)  # memory_s = 2


def test_reload_on_empty_magazine():
    agent = make_agent()
    # the HUD (OCR) still reads 0 for half a second after the reload ends, then 30
    frames = [{"objects": [], "text": hud(0)} for _ in range(3 * FPS)] + \
             [{"objects": [], "text": hud(30)} for _ in range(FPS)]
    d = feed(agent, frames)
    states = [x.state for x in d]
    assert states[0] == "EXPLORE" and states[1] == "RELOAD"  # needs two readings of 0 first
    assert d[1].action.reload
    assert sum(x.action.reload for x in d) == 1         # pressed once, not again on the stale 0
    assert states.count("RELOAD") == pytest.approx(2.5 * FPS, abs=2)
    assert states[-1] == "EXPLORE"


def test_low_ammo_waits_while_enemy_close():
    agent = make_agent()
    feed(agent, [{"objects": [person(1, 6.0)]}] * 4)
    d = feed(agent, [{"objects": [person(1, 6.0)], "text": hud(3)} for _ in range(5)], start=4)
    assert all(x.state.startswith("ENGAGE") for x in d)  # 3 rounds left, enemy close: keep fighting
    d = feed(agent, [{"objects": [], "text": hud(3)} for _ in range(40)], start=9)
    assert "RELOAD" in [x.state for x in d]


def test_single_stray_ammo_reading_ignored_and_old_reading_expires():
    b = WorldBuilder(W, H, FPS)
    assert b.update(0, {"text": hud(4)}).me.ammo is None          # read once: not trusted
    assert b.update(1, {"text": hud(7)}).me.ammo is None          # different value: starts over
    assert b.update(2, {"text": hud(7)}).me.ammo == 7             # same value twice: confirmed
    held = b.update(3, {"text": []})                              # OCR frame with nothing in the corner
    assert held.me.ammo == 7
    assert b.update(3 + 3 * FPS + 1, {"text": []}).me.ammo is None  # not read again for 3 s: forgotten


def test_brief_dropout_stays_in_engage():
    agent = make_agent()
    feed(agent, [{"objects": [person(1, 6.0)]}] * 10)
    gap = feed(agent, [{"objects": []}] * 10, start=10)          # 1/6 s without a detection
    assert all(x.state.startswith("ENGAGE") for x in gap)
    back = feed(agent, [{"objects": [person(1, 6.0)]}], start=20)
    assert back[0].state == "ENGAGE.FIRE" and back[0].target == 1


def test_explore_turns_away_from_wall():
    depth = np.full((100, 100), 0.1, dtype=np.float32)
    depth[30:70, 38:62] = 1.0      # close in the centre
    depth[30:70, 62:85] = 0.9      # close on the right too
    agent = make_agent()
    d = feed(agent, [{"objects": [], "depth": depth}])
    assert d[0].action.forward == 0.0 and d[0].action.yaw < 0  # turns left
    assert "blocked" in d[0].reason


def test_dwell_time_before_lower_priority():
    fsm = FpsStateMachine(FPS, cfg=AgentConfig(min_state_frames=6))
    b = WorldBuilder(W, H, FPS, WorldConfig(memory_s=0.0))
    for i in range(3):
        fsm.step(b.update(i, {"objects": [person(1, 8.0)]}))
    assert fsm.mode == "ENGAGE"
    d = fsm.step(b.update(3, {"objects": []}))       # target gone, nothing remembered
    assert d.state.startswith("ENGAGE") and "holding" in d.reason
    states = [fsm.step(b.update(i, {"objects": []})).state for i in range(4, 10)]
    assert states == ["ENGAGE"] * 4 + ["EXPLORE"] * 2  # ENGAGE (from frame 2) lasts 6 frames


# --- pipeline, log, overlay ---------------------------------------------------------------
class FakeObjects(Stage):
    name = "objects"

    def setup(self, info):
        self.info = info

    def process(self, index, frame):
        return [{"box": (20, 10, 30, 40), "id": 1, "conf": 0.9}] if index >= 10 else []


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


def test_pipeline_runs_agent_logs_and_times_it(clip, tmp_path):
    log = tmp_path / "actions.jsonl"
    agent = DecisionAgent(log_path=log, world_cfg=WorldConfig(min_box_h_px=4))
    seen = []
    result = pipeline.run(VideoReader(clip), [FakeObjects()], probe(clip), warmup=0, sync=False, agents=[agent],
                          on_frame=lambda i, f, latest: seen.append(latest["agent"]["decision"].state))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 30 and records[0]["frame"] == 0
    assert records[0]["state"] == "EXPLORE" and records[-1]["state"].startswith("ENGAGE")
    assert set(records[-1]["action"]) == {"move", "look", "fire", "ads", "reload", "crouch", "jump", "aim"}
    assert "move" in records[0]["action"] and "aim" not in records[0]["action"]  # no target, no aim point
    assert seen == [r["state"] for r in records]
    assert "agent" in {r["stage"] for r in report.summarize(result)}


def test_overlay_draws_agent():
    agent = make_agent()
    out = feed(agent, [{"objects": [person(1, 6.0, yaw_deg=5.0)]}] * 4)
    results = {"objects": [person(1, 6.0, yaw_deg=5.0)],
               "agent": {"decision": out[-1], "world": agent.builder.update(4, {"objects": [person(1, 6.0, yaw_deg=5.0)]})}}
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    img = draw(frame, results)
    assert img.any() and not frame.any()


# --- depth distance, closest policy, snap aim, coasting -------------------------------------
def test_depth_distance_fixes_crouched_target():
    b = WorldBuilder(W, H, FPS)
    standing = [(person(1, 8.0, yaw_deg=-15), 8.0), (person(2, 16.0, yaw_deg=15), 16.0)]
    low = crouched(3, 8.0, yaw_deg=0)
    for i in range(30):
        objs = standing + [(low, 8.0)]
        world = b.update(i, {"objects": [o for o, _ in objs], "depth": depth_for(objs)})
    t = {x.id: x for x in world.targets}
    assert b.depth_scale == pytest.approx(100.0, rel=0.05)       # calibrated from the standing two
    assert t[3].dist_box_m == pytest.approx(16.0, rel=0.02)       # short box: looks twice as far
    assert t[3].distance_m == pytest.approx(8.0, rel=0.1)         # depth gets it right
    assert t[2].distance_m == pytest.approx(16.0, rel=0.1)


def test_box_distance_without_depth_or_calibration():
    b = WorldBuilder(W, H, FPS)
    low = crouched(1, 8.0)
    for i in range(3):  # only a crouched target: nothing to calibrate the depth scale from
        world = b.update(i, {"objects": [low], "depth": depth_for([(low, 8.0)])})
    assert b.depth_scale is None
    assert world.targets[0].distance_m == world.targets[0].dist_box_m


def test_closest_policy_ignores_crosshair_and_switches_only_when_clearly_closer():
    agent = make_agent()
    d = feed(agent, [{"objects": [person(1, 12.0, yaw_deg=0.5), person(2, 5.0, yaw_deg=25.0)]}] * 4)
    assert d[-1].target == 2                                      # closest, though far from the crosshair
    d = feed(agent, [{"objects": [person(2, 5.0, yaw_deg=25.0), person(3, 4.5, yaw_deg=0.0)]}] * 4, start=4)
    assert d[-1].target == 2                                      # only 10% closer: keep the current one
    d = feed(agent, [{"objects": [person(2, 5.0, yaw_deg=25.0), person(3, 3.0, yaw_deg=0.0)]}] * 2, start=8)
    assert d[-1].target == 3                                      # 40% closer: switch


def test_score_policy_still_available():
    agent = make_agent(target_policy="score")
    d = feed(agent, [{"objects": [person(1, 12.0, yaw_deg=0.5), person(2, 9.0, yaw_deg=35.0)]}] * 4)
    assert d[-1].target == 1                                      # centred one wins on score


def test_snap_shoots_at_coordinates_in_one_tick():
    agent = make_agent(aim_mode="snap")
    d = feed(agent, [{"objects": [person(1, 6.0, yaw_deg=10.0, pitch_deg=2.0)]}] * 4)
    last = d[-1]
    assert last.state == "ENGAGE.FIRE" and last.action.fire
    assert last.action.yaw == pytest.approx(10.0, abs=0.05) and last.action.pitch == pytest.approx(2.0, abs=0.05)
    x, y = last.action.aim_px
    assert x == pytest.approx(W / 2 + FX * math.tan(math.radians(10.0)), abs=1)
    assert last.reason.startswith("shoot #1") and f"({x:.0f}, {y:.0f})" in last.reason
    log = last.action.to_dict()["aim"]
    assert log["deg"] == [pytest.approx(10.0, abs=0.05), pytest.approx(2.0, abs=0.05)]


def test_snap_too_far_off_turns_first():
    agent = make_agent(aim_mode="snap", max_snap_deg=30.0)
    d = feed(agent, [{"objects": [person(1, 6.0, yaw_deg=38.0)]}] * 4)
    assert d[-1].state == "ENGAGE.ACQUIRE" and not d[-1].action.fire and 0 < d[-1].action.yaw < 38.0


def test_short_gap_keeps_aim_and_burst():
    agent = make_agent(aim_mode="snap")
    feed(agent, [{"objects": [person(1, 6.0, yaw_deg=5.0)]}] * 6)
    gap = feed(agent, [{"objects": []}] * 6, start=6)
    assert all(x.state == "ENGAGE.FIRE" for x in gap)
    assert all(x.action.yaw == pytest.approx(5.0, abs=0.05) for x in gap)  # still aiming at the last position
    assert any(x.action.fire for x in gap) and "out of view" in gap[0].reason


# --- camera motion: remembered targets follow the view ----------------------------------------
def textured(seed=0, size=(H, W + 400)):
    """A random textured scene wider than the frame, to slide the 'camera' over."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, size, dtype=np.uint8), (0, 0), 3)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_motion_stage_measures_camera_turn():
    from gpp.adapters.motion import MotionStage
    stage = MotionStage(hfov_deg=80.0)
    stage.setup(INFO)
    scene = textured()
    first = stage.process(0, scene[:, 200:200 + W])
    assert not first["valid"]
    out = stage.process(1, scene[:, 230:230 + W])        # view slid 30 px right = camera turned right
    assert out["valid"]
    assert out["dx"] == pytest.approx(-30.0, abs=1.5)
    assert out["yaw"] == pytest.approx(math.degrees(math.atan(30.0 / FX)), abs=0.1)
    assert out["pitch"] == pytest.approx(0.0, abs=0.2)      # sub-pixel noise of phase correlation


def test_lost_target_follows_camera():
    b = WorldBuilder(W, H, FPS)
    for i in range(3):
        b.update(i, {"objects": [person(1, 8.0, yaw_deg=10.0)], "motion": {"yaw": 0.0, "pitch": 0.0, "valid": True}})
    for i in range(3, 13):  # target gone; camera turns 2° right per frame
        world = b.update(i, {"objects": [], "motion": {"yaw": 2.0, "pitch": 0.5, "valid": True}})
    (t,) = world.targets
    assert not t.visible and world.has_motion
    assert t.yaw_err == pytest.approx(-10.0, abs=0.01)     # was 10° right, view turned 20° right
    assert t.pitch_err == pytest.approx(-5.0, abs=0.01)
    assert t.aim[0] == pytest.approx(W / 2 + FX * math.tan(math.radians(-10.0)), abs=0.5)
    x1, y1, x2, y2 = t.box
    assert (x1 + x2) / 2 == pytest.approx(t.aim[0], abs=0.5)  # box moved with it


def test_same_motion_result_not_applied_twice():
    b = WorldBuilder(W, H, FPS)
    for i in range(3):
        b.update(i, {"objects": [person(1, 8.0)]})
    m = {"yaw": 5.0, "pitch": 0.0, "valid": True}
    b.update(3, {"objects": [], "motion": m})
    world = b.update(4, {"objects": [], "motion": m})       # stage didn't run: same object
    assert world.targets[0].yaw_err == pytest.approx(-5.0)


def test_search_turns_toward_followed_target_and_stops_on_it():
    agent = make_agent()
    still = lambda: {"yaw": 0.0, "pitch": 0.0, "valid": True}
    feed(agent, [{"objects": [person(1, 8.0, yaw_deg=-20.0)], "motion": still()} for _ in range(10)])
    # target gone; the agent turns left and (closed loop) the camera follows its command
    frames, d = [], []
    turn = 0.0
    for i in range(10, 10 + FPS):
        r = agent.process(i, {"objects": [], "motion": {"yaw": turn, "pitch": 0.0, "valid": True}})["decision"]
        turn = r.action.yaw
        d.append(r)
    # it turns left onto where the target was (already while coasting through the grace period)
    assert d[0].action.yaw < 0
    assert sum(x.action.yaw for x in d) == pytest.approx(-20.0, abs=2.0)
    search = [x for x in d if x.state == "SEARCH"]
    assert search and "watching" in search[-1].reason          # then watches that spot
    assert abs(agent.builder.tracks[1].yaw_err) <= 2.0


# --- NPC movement: velocity, carrying lost targets forward, lead --------------------------------
def moving(builder_or_agent, frames, rate_dps=30.0, start_yaw=-10.0, camera_dps=0.0, start=0):
    """Feed a target moving right at rate_dps (world) while the camera turns right at camera_dps."""
    out = []
    for i in range(frames):
        t = (start + i) / FPS
        on_screen = start_yaw + (rate_dps - camera_dps) * t
        r = {"objects": [person(1, 8.0, yaw_deg=on_screen)], "motion": {"yaw": camera_dps / FPS, "pitch": 0.0, "valid": True}}
        out.append(builder_or_agent.update(start + i, r) if isinstance(builder_or_agent, WorldBuilder)
                   else builder_or_agent.process(start + i, r)["decision"])
    return out


def test_velocity_is_world_motion_not_screen_motion():
    b = WorldBuilder(W, H, FPS)
    world = moving(b, 30, rate_dps=30.0, camera_dps=20.0)[-1]   # on screen only 10°/s
    assert world.targets[0].yaw_rate == pytest.approx(30.0, abs=1.0)
    assert world.targets[0].rate_samples >= 3


def test_lost_target_keeps_moving_for_a_while():
    b = WorldBuilder(W, H, FPS)
    world = moving(b, 30, rate_dps=30.0)[-1]
    last = world.targets[0].yaw_err
    still = {"yaw": 0.0, "pitch": 0.0, "valid": True}
    for i in range(30, 30 + FPS):  # gone for 1 s, camera still
        world = b.update(i, {"objects": [], "motion": still})
        if i == 30 + FPS // 4 - 1:
            assert world.targets[0].yaw_err == pytest.approx(last + 30.0 * 0.25, abs=0.6)
    assert world.targets[0].yaw_err == pytest.approx(last + 30.0 * 0.5, abs=0.6)  # stops after predict_s


def test_lead_aims_ahead_of_moving_target():
    agent = make_agent(aim_mode="snap", lead_s=0.1)
    d = moving(agent, 30, rate_dps=30.0, start_yaw=-1.0)
    track = agent.builder.tracks[1]
    assert d[-1].action.yaw == pytest.approx(track.yaw_err + 3.0, abs=0.3)  # 30°/s x 0.1 s ahead
    assert "lead +3.0°" in d[-1].reason


def test_no_lead_for_still_target():
    agent = make_agent(aim_mode="snap", lead_s=0.1)
    d = feed(agent, [{"objects": [person(1, 8.0, yaw_deg=4.0)]}] * 10)
    assert d[-1].action.yaw == pytest.approx(4.0, abs=0.05) and "lead" not in d[-1].reason


# --- health from the damage vignette -----------------------------------------------------------
def scene(edge_red=0.0, centre_red=0.0, size=(270, 480)):
    """A grey frame with a share of red pixels at the edges and in the centre (1080p)."""
    h, w = size
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    rng = np.random.default_rng(1)
    ring = np.ones((h, w), bool)
    ring[int(.15 * h):int(.85 * h), int(.15 * w):int(.85 * w)] = False
    for mask, share in ((ring, edge_red), (~ring, centre_red)):
        pick = mask & (rng.random((h, w)) < share)
        img[pick] = (20, 20, 200)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)


def health_stage(**kw):
    from gpp.adapters.health import HealthStage
    s = HealthStage(**kw)
    s.setup(INFO)
    return s


def test_health_drops_on_red_edges_not_on_red_scene():
    s = health_stage()
    quiet = s.process(0, scene())
    assert quiet["health"] == pytest.approx(1.0) and not quiet["hit"]
    hit = s.process(2, scene(edge_red=0.05))
    assert hit["hit"] and hit["damage"] == pytest.approx((0.05 - 0.004) / 0.06, abs=0.06) and hit["health"] < 0.3
    s2 = health_stage()
    red_room = s2.process(0, scene(edge_red=0.3, centre_red=0.3))   # red everywhere: a red scene
    assert red_room["health"] == pytest.approx(1.0) and not red_room["hit"]


def arcs(share, dot=True):
    """Grey 1080p frame with red pixels in the annulus around the crosshair, plus a red-dot sight."""
    h, w = 270, 480
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2) / h
    rng = np.random.default_rng(2)
    img[(r > 0.12) & (r < 0.3) & (rng.random((h, w)) < share)] = (20, 20, 200)
    if dot:
        img[r < 0.03] = (20, 20, 220)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)


def test_hit_direction_arcs_count_and_red_dot_does_not():
    s = health_stage()
    assert not s.process(0, arcs(0.0))["hit"]                     # red-dot sight alone: no hit
    out = s.process(2, arcs(0.04))
    assert out["hit"] and out["damage"] > 0.5


def test_health_drops_fast_recovers_slowly():
    s = health_stage(recover_s=1.5, every_n=2)
    s.process(0, scene(edge_red=0.07))
    assert s.health == pytest.approx(0.0, abs=0.01)
    for i in range(1, 46):                                         # 1.5 s at 60 fps / 2
        out = s.process(2 * i, scene())
    assert out["health"] == pytest.approx(1 - math.exp(-1), abs=0.05)


def test_recover_when_hurt_until_health_back():
    agent = make_agent()
    hp = lambda v: {"health": v}
    d = feed(agent, [{"objects": [person(1, 6.0)], "health": hp(1.0)}] * 4)
    assert d[-1].state.startswith("ENGAGE")
    d = feed(agent, [{"objects": [person(1, 6.0)], "health": hp(0.2)}] * 2, start=4)
    assert d[-1].state == "RECOVER" and d[-1].action.forward == -1.0 and not d[-1].action.fire
    assert "health 20%" in d[-1].reason
    d = feed(agent, [{"objects": [person(1, 6.0)], "health": hp(0.5)}] * 10, start=6)
    assert all(x.state == "RECOVER" for x in d)                    # 50% isn't back above 60% yet
    d = feed(agent, [{"objects": [person(1, 6.0)], "health": hp(0.7)}] * 10, start=16)
    assert d[-1].state.startswith("ENGAGE")


def test_recover_on_any_hit_for_two_seconds():
    agent = make_agent()
    hit = {"health": 0.98, "hit": True}
    feed(agent, [{"objects": [person(1, 6.0)], "health": {"health": 1.0, "hit": False}}] * 4)
    d = feed(agent, [{"objects": [person(1, 6.0)], "health": hit}], start=4)       # one small hit
    assert d[0].state == "RECOVER" and not d[0].action.fire and "hit 0 ms ago" in d[0].reason
    calm = [{"objects": [person(1, 6.0)], "health": {"health": 0.99, "hit": False}} for _ in range(3 * FPS)]
    d = feed(agent, calm, start=5)
    states = [x.state for x in d]
    assert states.index(next(x for x in states if x != "RECOVER")) == pytest.approx(2.0 * FPS, abs=2)
    assert states[-1].startswith("ENGAGE")


def test_recover_on_hit_can_be_turned_off():
    agent = make_agent(recover_on_hit=False)
    feed(agent, [{"objects": [person(1, 6.0)], "health": {"health": 1.0, "hit": False}}] * 4)
    d = feed(agent, [{"objects": [person(1, 6.0)], "health": {"health": 0.9, "hit": True}}], start=4)
    assert d[0].state.startswith("ENGAGE")                         # 90% is above low_health

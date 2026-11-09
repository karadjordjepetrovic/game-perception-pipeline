"""Decision agent: turns the perception results into FPS input, one decision per frame.

    results ──► WorldBuilder (gpp.world) ──► FpsStateMachine (fsm.py) ──► Decision ──► JSONL log

DecisionAgent puts the three together behind the same setup / process / close shape as a
Stage, but process() takes the latest results of all stages instead of the frame.
"""
import json
from pathlib import Path

from ..world import WorldBuilder, WorldConfig
from .actions import Action
from .fsm import AgentConfig, Decision, FpsStateMachine


class DecisionAgent:
    name = "agent"
    timed = True

    def __init__(self, log_path=None, world_cfg=None, agent_cfg=None):
        self.log_path = Path(log_path) if log_path else None
        self.world_cfg = world_cfg or WorldConfig()
        self.agent_cfg = agent_cfg or AgentConfig()
        self.log = None

    def setup(self, info):
        self.builder = WorldBuilder(info.width, info.height, info.fps, self.world_cfg)
        self.fsm = FpsStateMachine(info.fps, hfov_deg=self.world_cfg.hfov_deg, cfg=self.agent_cfg,
                                   assume_foe=self.world_cfg.assume_foe)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = open(self.log_path, "w", encoding="utf-8")

    def process(self, index, results):
        world = self.builder.update(index, results)
        decision = self.fsm.step(world)
        if self.log:
            record = {"frame": index, "t": round(world.t, 3), **decision.to_dict(),
                      "world": {"targets": [t.to_dict() for t in world.targets], "ammo": world.me.ammo,
                                "blocked": {k: round(v, 2) for k, v in world.me.blocked.items()}}}
            self.log.write(json.dumps(record) + "\n")
        return {"decision": decision, "world": world}

    def close(self):
        if self.log:
            self.log.close()
            self.log = None


__all__ = ["Action", "AgentConfig", "Decision", "DecisionAgent", "FpsStateMachine", "WorldConfig"]

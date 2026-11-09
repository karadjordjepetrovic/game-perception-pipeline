"""FPS input for one tick (one video frame): the same shape a gamepad or mouse/keyboard driver would take."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Action:
    forward: float = 0.0     # -1 (back) .. 1 (forward)
    strafe: float = 0.0      # -1 (left) .. 1 (right)
    yaw: float = 0.0         # degrees to turn this tick, + = right
    pitch: float = 0.0       # degrees to turn this tick, + = up
    fire: bool = False
    ads: bool = False        # aim down sights
    reload: bool = False
    crouch: bool = False
    jump: bool = False
    aim_px: Optional[tuple] = None   # target's aim point on screen (x, y), when aiming at one
    aim_deg: Optional[tuple] = None  # its offset from the crosshair (yaw + right, pitch + up)
    move_to: Optional[tuple] = None  # where it's walking to, metres (x right, z forward), e.g. a hiding spot

    def to_dict(self):
        d = {"move": [round(self.forward, 2), round(self.strafe, 2)],
                "look": [round(self.yaw, 2), round(self.pitch, 2)],
                "fire": self.fire, "ads": self.ads, "reload": self.reload,
                "crouch": self.crouch, "jump": self.jump}
        if self.aim_px is not None:
            d["aim"] = {"px": [round(self.aim_px[0]), round(self.aim_px[1])],
                        "deg": [round(self.aim_deg[0], 2), round(self.aim_deg[1], 2)]}
        if self.move_to is not None:
            d["move_to"] = [round(self.move_to[0], 2), round(self.move_to[1], 2)]
        return d

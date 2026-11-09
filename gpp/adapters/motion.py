"""Camera motion: how far the view turned since the previous frame (yaw / pitch, degrees).

Phase correlation (cv2.phaseCorrelate) between consecutive frames finds the dominant image
shift. In a first-person game that shift is mostly the camera turning: rotation moves the
whole scene the same way, while walking only adds parallax. The shift is converted to
degrees with the same pinhole camera as gpp.world (fx from the horizontal FOV).

Only a band of the frame is used: the HUD (minimap, ammo, crosshair) and the weapon
viewmodel don't move with the camera and would pull the estimate toward zero. CPU only,
about a millisecond on a downscaled band, so no GPU needed.

Result per frame: {"yaw": deg (+ = turned right), "pitch": deg (+ = turned up),
"dx", "dy": shift in frame pixels, "response": 0..1 peak strength, "valid": bool}.
"""
import math

import cv2
import numpy as np

from ..stage import Stage


class MotionStage(Stage):
    name = "motion"

    def __init__(self, hfov_deg=80.0, band=(0.10, 0.10, 0.90, 0.55), width=384, min_response=0.05):
        self.hfov_deg = hfov_deg
        self.band = band              # (x1, y1, x2, y2) as fractions of the frame
        self.width = width            # band is downscaled to this width
        self.min_response = min_response
        self.prev = None

    def setup(self, info):
        self.fx = (info.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        x1, y1, x2, y2 = self.band
        self.crop = (int(x1 * info.width), int(y1 * info.height), int(x2 * info.width), int(y2 * info.height))
        cw, ch = self.crop[2] - self.crop[0], self.crop[3] - self.crop[1]
        self.scale = self.width / cw  # downscale factor
        self.size = (self.width, max(16, int(round(ch * self.scale))))
        self.window = cv2.createHanningWindow(self.size, cv2.CV_32F)
        self.prev = None

    def process(self, index, frame):
        x1, y1, x2, y2 = self.crop
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        cur = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        prev, self.prev = self.prev, cur
        if prev is None:
            return {"yaw": 0.0, "pitch": 0.0, "dx": 0.0, "dy": 0.0, "response": 0.0, "valid": False}
        (sx, sy), response = cv2.phaseCorrelate(prev, cur, self.window)
        dx, dy = sx / self.scale, sy / self.scale  # back to frame pixels
        valid = response >= self.min_response
        # camera turns right -> scene moves left (dx < 0); camera turns up -> scene moves down (dy > 0)
        yaw = -math.degrees(math.atan(dx / self.fx)) if valid else 0.0
        pitch = math.degrees(math.atan(dy / self.fx)) if valid else 0.0
        return {"yaw": yaw, "pitch": pitch, "dx": dx, "dy": dy, "response": float(response), "valid": bool(valid)}

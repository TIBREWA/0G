"""
court_viewer.py — Real-time top-down court map for Tennisbot Isaac Lab.

Renders a minimap showing robot position, FOV cone, ball memory,
live detections, zone counts, and odometry trail inside an Isaac Sim
omni.ui window using a ByteImageProvider for pixel rendering.

Usage: Automatically created by TennisbotEnv when running with GUI.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaac.tennisbot_env import TennisbotEnv

try:
    import omni.ui as ui
    HAS_UI = True
except ImportError:
    HAS_UI = False

# ── Court constants (must match env_cfg.py) ──────────────────
FENCE_HALF_X = 15.0   # m — half-length
FENCE_HALF_Y = 7.5    # m — half-width
NET_HALF_W = 6.35      # m — half-width of net
CAM_FOV = 78.0         # degrees
MAX_BALL_DIST = 6.0    # m

# ── Canvas layout ────────────────────────────────────────────
CANVAS_W = 400         # pixels (maps to court Y-axis)
CANVAS_H = 600         # pixels (maps to court X-axis)
MARGIN = 25            # pixels around the court
TRAIL_LEN = 200        # odometry trail length


# ── Color helpers (RGBA byte tuples) ─────────────────────────
def _blend(canvas, x, y, r, g, b, a=255):
    """Set a single pixel with alpha blending."""
    h, w = canvas.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        if a >= 255:
            canvas[y, x] = [r, g, b, 255]
        else:
            alpha = a / 255.0
            old = canvas[y, x]
            canvas[y, x] = [
                int(old[0] * (1 - alpha) + r * alpha),
                int(old[1] * (1 - alpha) + g * alpha),
                int(old[2] * (1 - alpha) + b * alpha),
                255,
            ]


def _draw_circle(canvas, cx, cy, radius, r, g, b, a=255, fill=True):
    """Draw a circle (filled or outline) with alpha."""
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d2 = dx * dx + dy * dy
            if fill:
                if d2 <= radius * radius:
                    _blend(canvas, cx + dx, cy + dy, r, g, b, a)
            else:
                if abs(d2 - radius * radius) <= radius * 2:
                    _blend(canvas, cx + dx, cy + dy, r, g, b, a)


def _draw_line(canvas, x0, y0, x1, y1, r, g, b, a=255):
    """Draw a line using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        _blend(canvas, x0, y0, r, g, b, a)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_rect(canvas, x0, y0, x1, y1, r, g, b, a=255, fill=True):
    """Draw a rectangle (filled or outline)."""
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    if fill:
        h, w = canvas.shape[:2]
        x0c, x1c = max(0, x0), min(w, x1 + 1)
        y0c, y1c = max(0, y0), min(h, y1 + 1)
        if a >= 255:
            canvas[y0c:y1c, x0c:x1c] = [r, g, b, 255]
        else:
            alpha = a / 255.0
            region = canvas[y0c:y1c, x0c:x1c]
            canvas[y0c:y1c, x0c:x1c, 0] = (region[:, :, 0] * (1 - alpha) + r * alpha).astype(np.uint8)
            canvas[y0c:y1c, x0c:x1c, 1] = (region[:, :, 1] * (1 - alpha) + g * alpha).astype(np.uint8)
            canvas[y0c:y1c, x0c:x1c, 2] = (region[:, :, 2] * (1 - alpha) + b * alpha).astype(np.uint8)
    else:
        _draw_line(canvas, x0, y0, x1, y0, r, g, b, a)
        _draw_line(canvas, x1, y0, x1, y1, r, g, b, a)
        _draw_line(canvas, x1, y1, x0, y1, r, g, b, a)
        _draw_line(canvas, x0, y1, x0, y0, r, g, b, a)


def _fill_triangle(canvas, pts, r, g, b, a=255):
    """Fill a triangle given 3 (x, y) points."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_y, max_y = max(0, min(ys)), min(canvas.shape[0] - 1, max(ys))
    for y in range(min_y, max_y + 1):
        intersections = []
        for i in range(3):
            j = (i + 1) % 3
            y0, y1 = pts[i][1], pts[j][1]
            x0, x1 = pts[i][0], pts[j][0]
            if y0 == y1:
                continue
            if min(y0, y1) <= y <= max(y0, y1):
                t = (y - y0) / (y1 - y0)
                ix = int(x0 + t * (x1 - x0))
                intersections.append(ix)
        if len(intersections) >= 2:
            intersections.sort()
            x_start = max(0, intersections[0])
            x_end = min(canvas.shape[1] - 1, intersections[-1])
            if a >= 255:
                canvas[y, x_start:x_end + 1] = [r, g, b, 255]
            else:
                alpha = a / 255.0
                region = canvas[y, x_start:x_end + 1]
                canvas[y, x_start:x_end + 1, 0] = (region[:, 0] * (1 - alpha) + r * alpha).astype(np.uint8)
                canvas[y, x_start:x_end + 1, 1] = (region[:, 1] * (1 - alpha) + g * alpha).astype(np.uint8)
                canvas[y, x_start:x_end + 1, 2] = (region[:, 2] * (1 - alpha) + b * alpha).astype(np.uint8)


def _draw_text_simple(canvas, x, y, text, r, g, b, scale=1):
    """Draw text using a minimal built-in 3x5 bitmap font. Scale multiplies size."""
    FONT = {
        '0': ["111", "101", "101", "101", "111"],
        '1': ["010", "110", "010", "010", "111"],
        '2': ["111", "001", "111", "100", "111"],
        '3': ["111", "001", "111", "001", "111"],
        '4': ["101", "101", "111", "001", "001"],
        '5': ["111", "100", "111", "001", "111"],
        '6': ["111", "100", "111", "101", "111"],
        '7': ["111", "001", "001", "001", "001"],
        '8': ["111", "101", "111", "101", "111"],
        '9': ["111", "101", "111", "001", "111"],
        'A': ["010", "101", "111", "101", "101"],
        'B': ["110", "101", "110", "101", "110"],
        'C': ["011", "100", "100", "100", "011"],
        'D': ["110", "101", "101", "101", "110"],
        'E': ["111", "100", "110", "100", "111"],
        'F': ["111", "100", "110", "100", "100"],
        'G': ["011", "100", "101", "101", "011"],
        'H': ["101", "101", "111", "101", "101"],
        'I': ["111", "010", "010", "010", "111"],
        'K': ["101", "110", "100", "110", "101"],
        'L': ["100", "100", "100", "100", "111"],
        'M': ["101", "111", "111", "101", "101"],
        'N': ["101", "111", "111", "111", "101"],
        'O': ["010", "101", "101", "101", "010"],
        'P': ["110", "101", "110", "100", "100"],
        'R': ["110", "101", "110", "110", "101"],
        'S': ["011", "100", "010", "001", "110"],
        'T': ["111", "010", "010", "010", "010"],
        'U': ["101", "101", "101", "101", "111"],
        'V': ["101", "101", "101", "101", "010"],
        'W': ["101", "101", "111", "111", "101"],
        'X': ["101", "101", "010", "101", "101"],
        'Y': ["101", "101", "010", "010", "010"],
        'Z': ["111", "001", "010", "100", "111"],
        '+': ["000", "010", "111", "010", "000"],
        '-': ["000", "000", "111", "000", "000"],
        '.': ["000", "000", "000", "000", "010"],
        ':': ["000", "010", "000", "010", "000"],
        '%': ["101", "001", "010", "100", "101"],
        '(': ["001", "010", "010", "010", "001"],
        ')': ["100", "010", "010", "010", "100"],
        '/': ["001", "001", "010", "100", "100"],
        '?': ["111", "001", "010", "000", "010"],
        ' ': ["000", "000", "000", "000", "000"],
    }
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch)
        if glyph is None:
            cx += 4 * scale
            continue
        for row_i, row in enumerate(glyph):
            for col_i, bit in enumerate(row):
                if bit == '1':
                    for sy in range(scale):
                        for sx in range(scale):
                            px = cx + col_i * scale + sx
                            py = y + row_i * scale + sy
                            _blend(canvas, px, py, r, g, b)
        cx += (len(glyph[0]) + 1) * scale


class CourtViewerWindow:
    """
    Top-down court minimap rendered into an omni.ui window.

    Shows robot position, FOV cone, ball memory map, live detections,
    zone counts, odometry trail, and stats overlay — all from the
    perspective of what the robot actually knows (sensor-realistic).
    """

    def __init__(self, env_index: int = 0):
        if not HAS_UI:
            return
        self._env_idx = env_index
        self._trail: list[tuple[float, float]] = []
        self._frame_counter = 0
        self._update_interval = 5  # only redraw every N steps
        self._canvas = np.zeros((CANVAS_H, CANVAS_W, 4), dtype=np.uint8)
        self._image_provider = ui.ByteImageProvider()
        self._canvas[:] = [20, 20, 25, 255]
        self._image_provider.set_bytes_data(
            self._canvas.flatten().tolist(),
            [CANVAS_W, CANVAS_H],
        )
        self._build()

    def _build(self):
        self._window = ui.Window(
            "Court Map",
            width=CANVAS_W + 20,
            height=CANVAS_H + 40,
            visible=True,
        )
        with self._window.frame:
            with ui.VStack():
                self._image_widget = ui.ImageWithProvider(
                    self._image_provider,
                    width=CANVAS_W,
                    height=CANVAS_H,
                    fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                )

    # ── Coordinate transform ────────────────────────────────

    def _world_to_px(self, wx: float, wy: float) -> tuple[int, int]:
        """World coords → pixel coords. X=court-length → vertical, Y=court-width → horizontal."""
        draw_w = CANVAS_W - 2 * MARGIN
        draw_h = CANVAS_H - 2 * MARGIN
        px = int(MARGIN + (wy + FENCE_HALF_Y) / (2 * FENCE_HALF_Y) * draw_w)
        py = int(MARGIN + (FENCE_HALF_X - wx) / (2 * FENCE_HALF_X) * draw_h)
        return px, py

    # ── Drawing layers ──────────────────────────────────────

    def _draw_court(self, canvas):
        """Court outline, zones, and net."""
        tl = self._world_to_px(-FENCE_HALF_X, -FENCE_HALF_Y)
        br = self._world_to_px(FENCE_HALF_X, FENCE_HALF_Y)
        _draw_rect(canvas, tl[0], tl[1], br[0], br[1], 25, 25, 30, fill=True)
        _draw_rect(canvas, tl[0], tl[1], br[0], br[1], 60, 60, 60, fill=False)

        # Quadrant lines (X=0 net line, Y=0 center line)
        nl = self._world_to_px(0, -FENCE_HALF_Y)
        nr = self._world_to_px(0, FENCE_HALF_Y)
        _draw_line(canvas, nl[0], nl[1], nr[0], nr[1], 255, 255, 255)

        # Net body (thicker white between -6.35 and +6.35)
        net_l = self._world_to_px(0, -NET_HALF_W)
        net_r = self._world_to_px(0, NET_HALF_W)
        for offset in range(-1, 2):
            _draw_line(canvas, net_l[0], net_l[1] + offset, net_r[0], net_r[1] + offset, 255, 255, 255)

        # Net gap indicators
        gap_l = self._world_to_px(0, -FENCE_HALF_Y)
        gap_ln = self._world_to_px(0, -NET_HALF_W)
        _draw_line(canvas, gap_l[0], gap_l[1], gap_ln[0], gap_ln[1], 80, 200, 80)
        gap_r = self._world_to_px(0, FENCE_HALF_Y)
        gap_rn = self._world_to_px(0, NET_HALF_W)
        _draw_line(canvas, gap_rn[0], gap_rn[1], gap_r[0], gap_r[1], 80, 200, 80)

        # Center line (Y=0)
        cl_t = self._world_to_px(-FENCE_HALF_X, 0)
        cl_b = self._world_to_px(FENCE_HALF_X, 0)
        for i in range(0, max(abs(cl_b[1] - cl_t[1]), 1), 6):
            t = i / max(abs(cl_b[1] - cl_t[1]), 1)
            px = int(cl_t[0] + t * (cl_b[0] - cl_t[0]))
            py = int(cl_t[1] + t * (cl_b[1] - cl_t[1]))
            _blend(canvas, px, py, 50, 50, 50)

        # Labels
        _draw_text_simple(canvas, MARGIN, MARGIN - 10, "SIDE B +X", 140, 140, 140)
        _draw_text_simple(canvas, MARGIN, CANVAS_H - MARGIN + 3, "SIDE A -X", 140, 140, 140)
        mid = self._world_to_px(0, 0)
        _draw_text_simple(canvas, mid[0] + 5, mid[1] - 8, "NET", 200, 200, 200)

    def _draw_trail(self, canvas):
        """Odometry trail as fading dots."""
        n = len(self._trail)
        for i, (tx, ty) in enumerate(self._trail):
            alpha = (i + 1) / max(n, 1)
            px, py = self._world_to_px(tx, ty)
            r = int(30 + 170 * alpha)
            g = int(150 + 80 * alpha)
            b = int(180 + 50 * alpha)
            rad = max(1, int(2 * alpha))
            _draw_circle(canvas, px, py, rad, r, g, b)

    def _draw_ball_map(self, canvas, env: TennisbotEnv):
        """Draw remembered balls from the ball memory map."""
        idx = self._env_idx
        valid = env._ball_map_valid[idx]  # (nb,)
        positions = env._ball_map[idx]    # (nb, 2)
        for b in range(valid.shape[0]):
            if valid[b]:
                bx = positions[b, 0].item()
                by = positions[b, 1].item()
                px, py = self._world_to_px(bx, by)
                _draw_circle(canvas, px, py, 3, 120, 120, 120)

    def _draw_balls_live(self, canvas, env: TennisbotEnv):
        """Draw balls currently detected in FOV (live) and ghost balls (unseen ground truth)."""
        idx = self._env_idx
        ball_pos = env._ball_pos[idx]    # (nb, 3)
        alive = env._ball_alive[idx]     # (nb,)
        nb = ball_pos.shape[0]

        # Compute which balls are in FOV for this env
        root_pos = env.robot.data.root_pos_w[idx] - env.scene.env_origins[idx]
        root_quat = env.robot.data.root_quat_w[idx]
        yaw = env._quat_to_yaw(root_quat.unsqueeze(0)).item()
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        fov_half = math.radians(CAM_FOV / 2.0)

        cam_x = root_pos[0].item() + env.cfg.camera_offset_x * cos_y
        cam_y = root_pos[1].item() + env.cfg.camera_offset_x * sin_y

        target_idx = -1
        target_dist = 1e6

        for b in range(nb):
            if not alive[b]:
                continue
            bx = ball_pos[b, 0].item()
            by = ball_pos[b, 1].item()
            dx = bx - cam_x
            dy = by - cam_y
            dist = math.hypot(dx, dy)
            local_x = dx * cos_y + dy * sin_y
            local_y = -dx * sin_y + dy * cos_y
            angle = math.atan2(local_y, local_x)

            px, py = self._world_to_px(bx, by)

            in_fov = abs(angle) <= fov_half and dist <= MAX_BALL_DIST
            if in_fov:
                if dist < target_dist:
                    target_dist = dist
                    target_idx = b
                # Yellow = live detection
                _draw_circle(canvas, px, py, 4, 0, 200, 255)
            else:
                # Check if in memory
                in_mem = False
                valid = env._ball_map_valid[idx]
                mem_pos = env._ball_map[idx]
                for m in range(valid.shape[0]):
                    if valid[m]:
                        mdist = math.hypot(bx - mem_pos[m, 0].item(), by - mem_pos[m, 1].item())
                        if mdist < 0.4:
                            in_mem = True
                            break
                if not in_mem:
                    # Ghost ball — faint ring
                    _draw_circle(canvas, px, py, 4, 45, 45, 50, fill=False)

        # Highlight target
        if target_idx >= 0:
            bx = ball_pos[target_idx, 0].item()
            by = ball_pos[target_idx, 1].item()
            px, py = self._world_to_px(bx, by)
            _draw_circle(canvas, px, py, 6, 255, 60, 60)
            _draw_circle(canvas, px, py, 7, 255, 120, 120, fill=False)

    def _draw_robot(self, canvas, env: TennisbotEnv):
        """Robot body, heading arrow, and FOV cone."""
        idx = self._env_idx
        root_pos = env.robot.data.root_pos_w[idx] - env.scene.env_origins[idx]
        root_quat = env.robot.data.root_quat_w[idx]
        yaw = env._quat_to_yaw(root_quat.unsqueeze(0)).item()

        rx, ry = self._world_to_px(root_pos[0].item(), root_pos[1].item())

        # FOV cone
        fov_half = math.radians(CAM_FOV / 2.0)
        cone_len_px = 55
        pixel_yaw = -(yaw - math.pi / 2)

        left_angle = pixel_yaw + fov_half
        right_angle = pixel_yaw - fov_half

        lx = int(rx + cone_len_px * math.cos(left_angle))
        ly = int(ry - cone_len_px * math.sin(left_angle))
        rx2 = int(rx + cone_len_px * math.cos(right_angle))
        ry2 = int(ry - cone_len_px * math.sin(right_angle))

        # Semi-transparent cone
        _fill_triangle(canvas, [(rx, ry), (lx, ly), (rx2, ry2)], 60, 180, 60, a=50)
        _draw_line(canvas, rx, ry, lx, ly, 100, 220, 100)
        _draw_line(canvas, rx, ry, rx2, ry2, 100, 220, 100)

        # Robot body
        _draw_circle(canvas, rx, ry, 6, 50, 200, 220)
        _draw_circle(canvas, rx, ry, 7, 255, 255, 255, fill=False)

        # Heading arrow
        arrow_len = 15
        ax = int(rx + arrow_len * math.cos(pixel_yaw))
        ay = int(ry - arrow_len * math.sin(pixel_yaw))
        _draw_line(canvas, rx, ry, ax, ay, 255, 255, 100)

    def _draw_zone_counts(self, canvas, env: TennisbotEnv):
        """Show ball count per quadrant from memory map."""
        idx = self._env_idx
        valid = env._ball_map_valid[idx]
        positions = env._ball_map[idx]
        total = valid.sum().item()
        if total == 0:
            return

        counts = [0, 0, 0, 0]
        for b in range(valid.shape[0]):
            if valid[b]:
                bx = positions[b, 0].item()
                by = positions[b, 1].item()
                if bx < 0 and by < 0:
                    counts[0] += 1
                elif bx < 0 and by >= 0:
                    counts[1] += 1
                elif bx >= 0 and by < 0:
                    counts[2] += 1
                else:
                    counts[3] += 1

        centers = [
            (-FENCE_HALF_X / 2, -FENCE_HALF_Y / 2),
            (-FENCE_HALF_X / 2, FENCE_HALF_Y / 2),
            (FENCE_HALF_X / 2, -FENCE_HALF_Y / 2),
            (FENCE_HALF_X / 2, FENCE_HALF_Y / 2),
        ]
        for i, (wx, wy) in enumerate(centers):
            if counts[i] > 0:
                px, py = self._world_to_px(wx, wy)
                pct = int(counts[i] / max(total, 1) * 100)
                _draw_text_simple(canvas, px - 10, py - 3, f"{counts[i]}({pct}%)", 200, 200, 200)

    def _draw_overlay(self, canvas, env: TennisbotEnv):
        """Stats panel at top."""
        idx = self._env_idx
        _draw_rect(canvas, 0, 0, CANVAS_W, 22, 15, 15, 20)

        # Balls remaining
        alive_count = env._ball_alive[idx].sum().item()
        collected = env._balls_collected_total[idx].item()
        _draw_text_simple(canvas, 5, 3, f"BALLS:{alive_count}", 50, 220, 240)

        # Collected
        _draw_text_simple(canvas, 90, 3, f"GOT:{collected}", 80, 220, 80)

        # Stuck
        stuck = env._is_stuck[idx].item() if hasattr(env._is_stuck[idx], 'item') else bool(env._is_stuck[idx])
        if stuck:
            _draw_text_simple(canvas, 170, 3, "STUCK", 240, 80, 80)
        else:
            _draw_text_simple(canvas, 170, 3, "OK", 80, 220, 80)

        # Ball side memory
        side = env._last_ball_side[idx].item()
        side_str = "L" if side < -0.5 else "R" if side > 0.5 else "?"
        _draw_text_simple(canvas, 230, 3, f"MEM:{side_str}", 140, 140, 140)

        # Search duration
        search = env._steps_since_detection[idx].item()
        if search > 10:
            _draw_text_simple(canvas, 300, 3, f"S:{int(search)}", 240, 180, 50)

        # Legend at bottom
        ly = CANVAS_H - 12
        _draw_circle(canvas, 10, ly, 3, 0, 200, 255)
        _draw_text_simple(canvas, 16, ly - 3, "LIVE", 180, 180, 180)
        _draw_circle(canvas, 60, ly, 3, 120, 120, 120)
        _draw_text_simple(canvas, 66, ly - 3, "MEM", 180, 180, 180)
        _draw_circle(canvas, 105, ly, 3, 45, 45, 50, fill=False)
        _draw_text_simple(canvas, 111, ly - 3, "UNSEEN", 120, 120, 120)
        _draw_circle(canvas, 170, ly, 3, 255, 60, 60)
        _draw_text_simple(canvas, 176, ly - 3, "TARGET", 180, 180, 180)

    # ── Public API ──────────────────────────────────────────

    def update(self, env: TennisbotEnv):
        """Redraw court map from current env state. Call after each step."""
        if not HAS_UI:
            return

        self._frame_counter += 1
        if self._frame_counter % self._update_interval != 0:
            return

        canvas = self._canvas
        canvas[:] = [20, 20, 25, 255]  # dark background

        idx = self._env_idx
        root_pos = env.robot.data.root_pos_w[idx] - env.scene.env_origins[idx]
        rx = root_pos[0].item()
        ry = root_pos[1].item()
        self._trail.append((rx, ry))
        if len(self._trail) > TRAIL_LEN:
            self._trail.pop(0)

        self._draw_court(canvas)
        self._draw_trail(canvas)
        self._draw_ball_map(canvas, env)
        self._draw_balls_live(canvas, env)
        self._draw_zone_counts(canvas, env)
        self._draw_robot(canvas, env)
        self._draw_overlay(canvas, env)

        # Push pixels to omni.ui
        self._image_provider.set_bytes_data(
            canvas.flatten().tolist(),
            [CANVAS_W, CANVAS_H],
        )

    def reset_trail(self):
        """Clear trail at episode start."""
        self._trail.clear()

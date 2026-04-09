"""
obs_monitor.py — Real-time observation monitor for Tennisbot Isaac Lab environment.

Displays all 42 observation dimensions plus reward info in an Isaac Sim UI window.
Automatically color-codes values: green=positive, red=negative, gray=near-zero.

Usage: Automatically created by TennisbotEnv when running with GUI (not --headless).
"""

from __future__ import annotations

try:
    import omni.ui as ui
    HAS_UI = True
except ImportError:
    HAS_UI = False

# ── Observation layout: (group_name, [(index, label, kind)]) ──
# kind: "bool" | "signed" | "unsigned"
OBS_GROUPS = [
    ("🎾 Ball Detection", [
        (0, "Detected", "bool"),
        (1, "Angle", "signed"),
        (2, "Distance", "unsigned"),
        (3, "Area", "unsigned"),
    ]),
    ("🤖 Robot Velocity", [
        (4, "Forward Vel (vx)", "signed"),
        (5, "Angular Vel (wz)", "signed"),
    ]),
    ("📊 Game State", [
        (6, "Balls Remaining", "unsigned"),
        (7, "Cmd Speed", "signed"),
        (8, "Speed Ratio", "signed"),
        (9, "Stuck Duration", "signed"),
    ]),
    ("📍 Odometry", [
        (10, "Odom X", "signed"),
        (11, "Odom Y", "signed"),
        (12, "Odom Yaw", "signed"),
    ]),
    ("🧠 Ball Memory", [
        (13, "Last Ball Side", "signed"),
        (14, "Nearby Left", "signed"),
        (15, "Nearby Right", "signed"),
    ]),
    ("🏐 Net", [
        (16, "Net Distance", "signed"),
        (17, "Cross Net", "bool"),
    ]),
    ("🧭 Nearest Zone", [
        (18, "Zone Angle", "signed"),
        (19, "Zone Distance", "unsigned"),
    ]),
    ("🗺️ Zone Grid (6x4)", [
        (20, "R0 C0 (-Y,-X)", "unsigned"),
        (21, "R0 C1", "unsigned"),
        (22, "R0 C2", "unsigned"),
        (23, "R0 C3", "unsigned"),
        (24, "R0 C4", "unsigned"),
        (25, "R0 C5 (-Y,+X)", "unsigned"),
        (26, "R1 C0", "unsigned"),
        (27, "R1 C1", "unsigned"),
        (28, "R1 C2", "unsigned"),
        (29, "R1 C3", "unsigned"),
        (30, "R1 C4", "unsigned"),
        (31, "R1 C5", "unsigned"),
        (32, "R2 C0", "unsigned"),
        (33, "R2 C1", "unsigned"),
        (34, "R2 C2", "unsigned"),
        (35, "R2 C3", "unsigned"),
        (36, "R2 C4", "unsigned"),
        (37, "R2 C5", "unsigned"),
        (38, "R3 C0 (+Y,-X)", "unsigned"),
        (39, "R3 C1", "unsigned"),
        (40, "R3 C2", "unsigned"),
        (41, "R3 C3", "unsigned"),
        (42, "R3 C4", "unsigned"),
        (43, "R3 C5 (+Y,+X)", "unsigned"),
    ]),
]

# ── Style constants ──
STYLE_TITLE = {"font_size": 18, "color": 0xFFFFCC44}
STYLE_GROUP = {"font_size": 14, "color": 0xFFDDDDDD}
STYLE_LABEL = {"font_size": 13, "color": 0xFFAAAAAA}
STYLE_VAL_POS = {"font_size": 14, "color": 0xFF44FF88}
STYLE_VAL_NEG = {"font_size": 14, "color": 0xFFFF6666}
STYLE_VAL_ZERO = {"font_size": 14, "color": 0xFF666666}
STYLE_VAL_BOOL_ON = {"font_size": 14, "color": 0xFF44DDFF}
STYLE_VAL_BOOL_OFF = {"font_size": 14, "color": 0xFF444444}
STYLE_REWARD_POS = {"font_size": 15, "color": 0xFF44FF44}
STYLE_REWARD_NEG = {"font_size": 15, "color": 0xFFFF4444}


class ObsMonitorWindow:
    """
    Real-time observation monitor window for Isaac Sim.

    Shows all 42 observation values grouped by category, plus live
    reward and episode statistics. Values are color-coded for quick reading.
    """

    def __init__(self, env_index: int = 0):
        """
        Args:
            env_index: Which environment to monitor (default: 0).
        """
        if not HAS_UI:
            return
        self._env_idx = env_index
        self._value_labels: dict[int, ui.Label] = {}
        self._reward_label = None
        self._episode_reward_label = None
        self._step_label = None
        self._balls_collected_label = None
        self._episode_reward = 0.0
        self._build()

    def _build(self):
        """Construct the omni.ui window layout."""
        self._window = ui.Window(
            "🎾 Tennisbot Obs Monitor",
            width=400,
            height=720,
            visible=True,
        )

        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            ):
                with ui.VStack(spacing=6, style={"margin": 10}):
                    # ── Title ──
                    ui.Label(
                        f"Monitoring Env #{self._env_idx}",
                        style=STYLE_TITLE,
                        height=28,
                    )
                    ui.Separator(height=2)

                    # ── Reward Section ──
                    with ui.CollapsableFrame("⭐ Reward", height=0):
                        with ui.VStack(spacing=4):
                            self._reward_label = self._make_row("Step Reward", "  0.0000")
                            self._episode_reward_label = self._make_row("Episode Total", "  0.0000")
                            self._step_label = self._make_row("Step #", "  0")
                            self._balls_collected_label = self._make_row("Balls Collected", "  0")
                            
                            ui.Separator(height=2)
                            ui.Label("  Breakdown:", style=STYLE_GROUP)
                            self._reward_cat_labels = {}
                            categories = [
                                "Step Pen", "Collect", "Approach (Vis)", "Loiter Pen", 
                                "Approach (Mem)", "Seek (Pop)", "Explore", "Stuck Pen", 
                                "Spin Pen", "Rev Pen", "Unstuck Bon", "All Done"
                            ]
                            for cat in categories:
                                self._reward_cat_labels[cat] = self._make_row(f"    {cat}", "  0.0000")

                    ui.Spacer(height=4)

                    # ── Action Commands ──
                    with ui.CollapsableFrame("🎮 Action Commands", height=0):
                        with ui.VStack(spacing=3):
                            self._cmd_left_label = self._make_row("Cmd Left (%)", "  0.00")
                            self._cmd_right_label = self._make_row("Cmd Right (%)", "  0.00")
                            self._cmd_direction_label = self._make_row("Direction", "  ---")
                            self._stuck_label = self._make_row("Stuck", "  NO")
                            self._stuck_counter_label = self._make_row("Stuck Counter", "  0")

                    ui.Spacer(height=4)

                    # ── Observation Groups ──
                    for group_name, entries in OBS_GROUPS:
                        with ui.CollapsableFrame(group_name, height=0):
                            with ui.VStack(spacing=3):
                                for idx, name, kind in entries:
                                    self._value_labels[idx] = self._make_row(
                                        f"[{idx:2d}] {name}", "  0.0000"
                                    )

    @staticmethod
    def _make_row(label_text: str, default_value: str) -> "ui.Label":
        """Create a single label + value row."""
        with ui.HStack(height=20):
            ui.Label(label_text, width=220, style=STYLE_LABEL)
            lbl = ui.Label(default_value, width=140, style=STYLE_VAL_ZERO)
        return lbl

    # ─────────────────────────────────────────────────────────
    # Live Updates
    # ─────────────────────────────────────────────────────────

    def update_obs(self, obs_tensor) -> None:
        """
        Update displayed observation values.

        Args:
            obs_tensor: (num_envs, 42) tensor from _get_observations.
        """
        if not HAS_UI or obs_tensor is None:
            return
        try:
            vals = obs_tensor[self._env_idx].detach().cpu().tolist()
            for group_name, entries in OBS_GROUPS:
                for idx, name, kind in entries:
                    if idx not in self._value_labels:
                        continue
                    val = vals[idx]
                    lbl = self._value_labels[idx]

                    if kind == "bool":
                        lbl.text = "  YES" if val > 0.5 else "  NO"
                        lbl.style = STYLE_VAL_BOOL_ON if val > 0.5 else STYLE_VAL_BOOL_OFF
                    else:
                        lbl.text = f"  {val:+.4f}"
                        if abs(val) < 0.01:
                            lbl.style = STYLE_VAL_ZERO
                        elif val > 0:
                            lbl.style = STYLE_VAL_POS
                        else:
                            lbl.style = STYLE_VAL_NEG
        except Exception:
            pass

    def update_reward(self, reward_tensor, reward_dict: dict, step: int, balls_collected: int = 0) -> None:
        """
        Update reward display.

        Args:
            reward_tensor: (num_envs,) reward tensor.
            reward_dict: Dictionary mapping string category names to (num_envs,) reward tensors.
            step: Current episode step count.
            balls_collected: Total balls collected this episode.
        """
        if not HAS_UI or reward_tensor is None:
            return
        try:
            r = reward_tensor[self._env_idx].item()
            self._episode_reward += r

            if self._reward_label:
                self._reward_label.text = f"  {r:+.4f}"
                self._reward_label.style = STYLE_REWARD_POS if r >= 0 else STYLE_REWARD_NEG

            if self._episode_reward_label:
                self._episode_reward_label.text = f"  {self._episode_reward:+.2f}"
                self._episode_reward_label.style = (
                    STYLE_REWARD_POS if self._episode_reward >= 0 else STYLE_REWARD_NEG
                )

            if self._step_label:
                self._step_label.text = f"  {step}"

            if self._balls_collected_label:
                self._balls_collected_label.text = f"  {balls_collected}"
                
            for cat, r_tensor in reward_dict.items():
                if cat in self._reward_cat_labels:
                    val = r_tensor[self._env_idx].item()
                    lbl = self._reward_cat_labels[cat]
                    lbl.text = f"  {val:+.4f}"
                    lbl.style = STYLE_REWARD_POS if val > 0.001 else (STYLE_REWARD_NEG if val < -0.001 else STYLE_VAL_ZERO)
        except Exception:
            pass

    def update_actions(self, env) -> None:
        """Update action command display from env state."""
        if not HAS_UI:
            return
        try:
            idx = self._env_idx
            cmd_l = env._last_cmd_left[idx].item()
            cmd_r = env._last_cmd_right[idx].item()
            stuck = env._is_stuck[idx].item() if hasattr(env._is_stuck[idx], 'item') else bool(env._is_stuck[idx])
            stuck_cnt = env._stuck_counter[idx].item()

            if self._cmd_left_label:
                self._cmd_left_label.text = f"  {cmd_l:+.1f}%"
                self._cmd_left_label.style = STYLE_VAL_POS if abs(cmd_l) > 1 else STYLE_VAL_ZERO
            if self._cmd_right_label:
                self._cmd_right_label.text = f"  {cmd_r:+.1f}%"
                self._cmd_right_label.style = STYLE_VAL_POS if abs(cmd_r) > 1 else STYLE_VAL_ZERO

            # Direction label
            if self._cmd_direction_label:
                if abs(cmd_l) < 5 and abs(cmd_r) < 5:
                    direction = "STOPPED"
                    style = STYLE_VAL_ZERO
                elif cmd_l < -5 and cmd_r < -5:
                    direction = "REVERSING"
                    style = STYLE_VAL_NEG
                elif abs(cmd_l - cmd_r) > 30:
                    direction = "SPINNING"
                    style = {"font_size": 14, "color": 0xFFFFAA44}
                else:
                    direction = "FORWARD"
                    style = STYLE_VAL_POS
                self._cmd_direction_label.text = f"  {direction}"
                self._cmd_direction_label.style = style

            if self._stuck_label:
                self._stuck_label.text = "  YES" if stuck else "  NO"
                self._stuck_label.style = STYLE_VAL_NEG if stuck else STYLE_VAL_POS
            if self._stuck_counter_label:
                self._stuck_counter_label.text = f"  {int(stuck_cnt)}"
                self._stuck_counter_label.style = STYLE_VAL_NEG if stuck_cnt > 5 else STYLE_VAL_ZERO
        except Exception:
            pass

    def reset_episode(self) -> None:
        """Reset episode-level tracking (call on env reset)."""
        self._episode_reward = 0.0

    def destroy(self) -> None:
        """Clean up the window."""
        if HAS_UI and hasattr(self, "_window"):
            self._window.visible = False

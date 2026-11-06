"""
TennisbotEnv — Isaac Lab DirectRLEnv for the tennis-ball-collecting rover.

GPU-accelerated replacement for rl/env.py (PyBullet).
All tensors are (num_envs, ...) on GPU.

Key design:
  - Ground-truth ball positions (no YOLO) with simulated camera FOV filtering.
  - Identical 22-dim observation space as PyBullet env for policy compatibility.
  - Ball "collection" = teleport below ground when in front of robot + close enough.
  - Obstacles are spawned as static rigid bodies at reset time.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG

from isaac.env_cfg import TennisbotEnvCfg


class TennisbotEnv(DirectRLEnv):
    """
    Isaac Lab environment for training a tennis-ball-collecting rover.

    Subclasses DirectRLEnv so the full step logic runs on GPU with torch tensors.
    """

    cfg: TennisbotEnvCfg

    def __init__(self, cfg: TennisbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Joint indices ────────────────────────────────────
        # The cylinder URDF has 2 driven front wheels + 1 free rear caster.
        driven_names = ["front_left_wheel_joint", "front_right_wheel_joint"]
        self._wheel_ids, _ = self.robot.find_joints(driven_names)
        self._left_id = self._wheel_ids[0]   # FL
        self._right_id = self._wheel_ids[1]  # FR

        # ── Per-env state buffers (on GPU) ───────────────────
        n = self.num_envs
        device = self.device

        # Ball state: (num_envs, num_balls, 3) positions; mask for "alive"
        self._ball_pos = torch.zeros(n, self.cfg.num_balls, 3, device=device)
        self._ball_alive = torch.ones(n, self.cfg.num_balls, dtype=torch.bool, device=device)

        # Obstacle state: positions and active mask (max_obstacles slots per env)
        max_obs = self.cfg.max_obstacles if self.cfg.spawn_obstacles else 0
        self._obstacle_local_pos = torch.zeros(n, max(max_obs, 1), 3, device=device)
        self._obstacle_active = torch.zeros(n, max(max_obs, 1), dtype=torch.bool, device=device)
        if self.cfg.spawn_obstacles:
            self._obstacle_sizes_t = torch.tensor(self._obstacle_half_extents, device=device)
            self._obstacle_heights_t = torch.tensor(self._obstacle_heights, device=device)

        # Stuck detection
        self._stuck_counter = torch.zeros(n, device=device)
        self._is_stuck = torch.zeros(n, dtype=torch.bool, device=device)
        self._was_stuck = torch.zeros(n, dtype=torch.bool, device=device)
        self._reverse_timer = torch.zeros(n, device=device)
        self._last_cmd_left = torch.zeros(n, device=device)
        self._last_cmd_right = torch.zeros(n, device=device)

        # Odometry (dead-reckoning from wheel encoders + IMU yaw)
        self._odom_x = torch.zeros(n, device=device)
        self._odom_y = torch.zeros(n, device=device)
        self._odom_yaw = torch.zeros(n, device=device)

        # Ball search memory
        self._last_ball_side = torch.zeros(n, device=device)
        self._last_ball_angle = torch.zeros(n, device=device)
        self._last_detected = torch.zeros(n, dtype=torch.bool, device=device)
        self._ball_cross_net = torch.zeros(n, device=device)
        self._spin_search_steps = torch.zeros(n, device=device)

        # Ball discovery map — starts empty, populated when balls enter FOV
        self._ball_discovered = torch.zeros(n, self.cfg.num_balls, dtype=torch.bool, device=device)
        self._ball_map_pos = torch.zeros(n, self.cfg.num_balls, 2, device=device)

        # Cached nearest-populated-zone direction (computed in obs, used in reward)
        self._zone_angle_norm = torch.zeros(n, device=device)
        self._zone_dist_norm = torch.zeros(n, device=device)

        # Previous distances for approach reward
        self._prev_nearest_dist = torch.full((n,), 1e6, device=device)
        self._prev_nearest_memory_dist = torch.full((n,), 1e6, device=device)

        # Loiter tracking
        self._loiter_counter = torch.zeros(n, device=device)

        # Zone exploration tracking
        self._zone_visited = torch.zeros(n, self.cfg.zone_grid_cols * self.cfg.zone_grid_rows, dtype=torch.bool, device=device)

        # Mission focus: flat zone index the policy is steered to, -1 = off (training default)
        self._focus_zone = torch.full((n,), -1, dtype=torch.long, device=device)

        # Balls collected per episode (for info tracking)
        self._balls_collected_total = torch.zeros(n, dtype=torch.int32, device=device)

        # Cached action values
        self._actions = torch.zeros(n, 2, device=device)

        # ── Observation Monitor GUI (only with GUI) ──────────
        self._obs_monitor = None
        try:
            from isaac.obs_monitor import ObsMonitorWindow, HAS_UI
            if HAS_UI:
                self._obs_monitor = ObsMonitorWindow(env_index=0)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # Scene Setup
    # ─────────────────────────────────────────────────────────

    def _setup_scene(self):
        """Spawn robot, ground plane, net, fences, and balls into the scene."""

        # ── Robot ────────────────────────────────────────────
        self.robot = Articulation(self.cfg.robot_cfg)

        # ── Ground plane (Visual Background + Physics) ───────
        ground_cfg = sim_utils.GroundPlaneCfg(
            size=(10000.0, 10000.0), # Expand to 10 kilometers to fill the void
            color=(0.1, 0.2, 0.15),  # Dark grassy green background
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                friction_combine_mode="multiply",  # was "max" — need "multiply" so caster friction works
                restitution=0.0,
            ),
        )
        ground_cfg.func("/World/ground", ground_cfg)

        # ── Low-friction material for rear caster ─────────────
        # In PyBullet the rear caster uses lateralFriction=0.2 so it can
        # slide sideways during in-place turns.  We replicate that here.
        self._apply_caster_low_friction()

        # ── Blender Visual Court ─────────────────────────────
        import os
        court_path = "/home/gui/PycharmProjects/pybullet/models/court/court.usdc"
        if os.path.exists(court_path):
            court_cfg = sim_utils.UsdFileCfg(usd_path=court_path, scale = (0.1,0.1,0.1))
            # Elevate by 1mm to prevent z-fighting with the ground plane
            court_cfg.func("/World/envs/env_0/court_visual", court_cfg, translation=(0.0, 0.0, 0.001))

        # ── Court boundaries (net + fences) as static colliders ──
        self._spawn_court_colliders()

        # ── Obstacles (kinematic rigid bodies) ───────────────
        # Create ONE shared high-friction physics material to avoid 64K material limit
        self._obstacle_objects = []
        if self.cfg.spawn_obstacles:
            from isaaclab.assets import RigidObjectCfg
            self._create_obstacle_material()
            # Use fixed sizes for all obstacles to ensure deterministic scene setup
            he, h = 0.5, 0.3
            self._obstacle_half_extents = [he] * self.cfg.max_obstacles
            self._obstacle_heights = [h] * self.cfg.max_obstacles

            for i in range(self.cfg.max_obstacles):
                obs_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/obstacle_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(he * 2, he * 2, h * 2),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.4, 0.4, 0.35),
                            opacity=1.0,
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
                )
                obs_obj = RigidObject(obs_cfg)
                self._obstacle_objects.append(obs_obj)
            # Bind the shared material to all obstacle prims
            self._bind_obstacle_material()

        # ── Clone environments ───────────────────────────────
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # ── Register robot ───────────────────────────────────
        self.scene.articulations["robot"] = self.robot

        # ── Register obstacles ───────────────────────────────
        for i, obs_obj in enumerate(self._obstacle_objects):
            self.scene.rigid_objects[f"obstacle_{i}"] = obs_obj

        # ── Lights ───────────────────────────────────────────
        light_cfg = sim_utils.DomeLightCfg(intensity=15000.0 * self.cfg.light_scale, color=(0.85, 0.85, 0.8))
        light_cfg.func("/World/Light", light_cfg)
        
        sun_cfg = sim_utils.DistantLightCfg(intensity=15000.0 * self.cfg.light_scale, color=(0.9, 0.9, 0.8))
        sun_cfg.func("/World/Sun", sun_cfg, translation=(1.0, 1.0, 10.0))
        
        # ── Visual Markers for Balls ─────────────────────────
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/balls"
        marker_cfg.markers["sphere"].radius = self.cfg.ball_radius
        marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.7, 0.9, 0.1) # Tennis ball green-yellow
        self.ball_markers = VisualizationMarkers(marker_cfg)

        # ── Visual Markers for FOV ─────────────────────────
        if self.cfg.debug_draw_fov:
            from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
            
            fov_visual_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/fov_lines",
                markers={
                    "line": sim_utils.CuboidCfg(
                        size=(self.cfg.max_ball_dist_m, 0.02, 0.02),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                    )
                }
            )
            self.fov_markers = VisualizationMarkers(fov_visual_cfg)
        else:
            self.fov_markers = None

    def _spawn_court_colliders(self):
        """Create the net and fence walls as static collision boxes."""
        # Net at x=0.2 (thin wall, 12.7m wide, 0.9m tall)
        net_cfg = sim_utils.CuboidCfg(
            size=(0.01, 12.7, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        )
        net_cfg.func("/World/envs/env_0/net", net_cfg, translation=(0.2, 0.0, 0.45))

        # Fences — 4 walls
        fence_specs = [
            ("fence_px", (15.0, 0.0, 1.5), (0.01, 15.0, 3.0)),   # +X
            ("fence_nx", (-15.0, 0.0, 1.5), (0.01, 15.0, 3.0)),  # -X
            ("fence_py", (0.0, 7.5, 1.5), (30.0, 0.01, 3.0)),    # +Y
            ("fence_ny", (0.0, -7.5, 1.5), (30.0, 0.01, 3.0)),   # -Y
        ]
        for name, pos, size in fence_specs:
            cfg = sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), metallic=0.5),
            )
            cfg.func(f"/World/envs/env_0/{name}", cfg, translation=pos)

    def _apply_caster_low_friction(self):
        """Apply low friction to the rear caster wheel so it can slide laterally during turns.

        In PyBullet this is done via:
            p.changeDynamics(robot_id, cj, lateralFriction=0.2,
                             spinningFriction=0.01, rollingFriction=0.001)

        Here we create a PhysX material with similar values and bind it to
        the rear caster's collision geometry.  This runs before clone_environments()
        so the material propagates to all envs automatically.
        """
        import omni.usd
        from pxr import UsdPhysics, UsdShade, Usd

        stage = omni.usd.get_context().get_stage()

        # ── Create a low-friction physics material ────────────
        mat_path = "/World/Materials/CasterLowFriction"
        mat_prim = stage.DefinePrim(mat_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        # With sphere collision, the contact is a single point that rolls freely
        # in any direction — lower friction is fine and won't cause spin-outs.
        phys_mat.CreateStaticFrictionAttr(0.35)
        phys_mat.CreateDynamicFrictionAttr(0.2)
        phys_mat.CreateRestitutionAttr(0.0)

        # ── Find the rear caster collision prim(s) and bind ───
        rear_link_path = "/World/envs/env_0/Robot/rear_wheel_link"
        rear_prim = stage.GetPrimAtPath(rear_link_path)
        if not rear_prim.IsValid():
            print(f"[WARN] Could not find rear caster at {rear_link_path}")
            return

        mat_shade = UsdShade.Material(mat_prim)
        bound = False
        for prim in Usd.PrimRange(rear_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                binding_api.Bind(mat_shade, UsdShade.Tokens.weakerThanDescendants, "physics")
                bound = True

        if not bound:
            # Fallback: bind to the link itself
            binding_api = UsdShade.MaterialBindingAPI.Apply(rear_prim)
            binding_api.Bind(mat_shade, UsdShade.Tokens.weakerThanDescendants, "physics")

        print("[INFO] Low-friction material applied to rear caster")

    def _create_obstacle_material(self):
        """Create a single shared high-friction physics material for all obstacles."""
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        mat_path = "/World/Materials/ObstacleHighFriction"
        mat_prim = stage.DefinePrim(mat_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        phys_mat.CreateStaticFrictionAttr(1.5)
        phys_mat.CreateDynamicFrictionAttr(1.5)
        phys_mat.CreateRestitutionAttr(0.0)
        print("[INFO] Shared obstacle high-friction material created")

    def _bind_obstacle_material(self):
        """Bind the shared obstacle material to all obstacle collision prims in env_0."""
        import omni.usd
        from pxr import UsdPhysics, UsdShade, Usd

        stage = omni.usd.get_context().get_stage()
        mat_prim = stage.GetPrimAtPath("/World/Materials/ObstacleHighFriction")
        if not mat_prim.IsValid():
            print("[WARN] Obstacle material not found, skipping binding")
            return

        mat_shade = UsdShade.Material(mat_prim)
        bound_count = 0
        for i in range(self.cfg.max_obstacles):
            obs_path = f"/World/envs/env_0/obstacle_{i}"
            obs_prim = stage.GetPrimAtPath(obs_path)
            if not obs_prim.IsValid():
                continue
            for prim in Usd.PrimRange(obs_prim):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding_api.Bind(mat_shade, UsdShade.Tokens.weakerThanDescendants, "physics")
                    bound_count += 1
        print(f"[INFO] Obstacle material bound to {bound_count} collision prims")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert [-1,1]² actions to wheel velocity targets."""
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._last_cmd_left = self._actions[:, 0] * 100.0   # percent
        self._last_cmd_right = self._actions[:, 1] * 100.0
        self._apply_action()

    def _apply_action(self) -> None:
        """Apply differential drive wheel velocities (called each physics substep)."""
        left_vel = self._actions[:, 0] * self.cfg.max_wheel_rad_s   # (num_envs,)
        right_vel = self._actions[:, 1] * self.cfg.max_wheel_rad_s

        # Build velocity targets for 2 driven wheels: [FL, FR]
        vel_targets = torch.zeros(self.num_envs, 2, device=self.device)
        vel_targets[:, 0] = left_vel    # FL
        vel_targets[:, 1] = right_vel   # FR

        self.robot.set_joint_velocity_target(vel_targets, joint_ids=self._wheel_ids)

    # ─────────────────────────────────────────────────────────
    # Observations (22-dim, identical to PyBullet env)
    # ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """Build the 22-dimensional observation tensor for all envs."""
        n = self.num_envs
        device = self.device

        # ── Robot state ──────────────────────────────────────
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins  # local pos
        root_quat = self.robot.data.root_quat_w  # (n, 4) wxyz
        root_lin_vel = self.robot.data.root_lin_vel_w  # (n, 3) world frame
        root_ang_vel = self.robot.data.root_ang_vel_w  # (n, 3) world frame

        # Extract yaw from quaternion (w, x, y, z)
        yaw = self._quat_to_yaw(root_quat)  # (n,)

        # Forward velocity in robot frame
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx = root_lin_vel[:, 0] * cos_yaw + root_lin_vel[:, 1] * sin_yaw  # (n,)
        wz = root_ang_vel[:, 2]  # (n,)

        # ── Update odometry (dead-reckoning with drift noise) ──
        # Integrate forward velocity and yaw rate instead of using ground-truth pose
        dt = self.cfg.sim.dt * self.cfg.decimation  # env step dt
        self._odom_yaw = self._odom_yaw + wz * dt
        # Add drift noise
        self._odom_yaw = self._odom_yaw + torch.randn(n, device=device) * self.cfg.odom_yaw_drift_std
        # Wrap to [-pi, pi]
        self._odom_yaw = torch.atan2(torch.sin(self._odom_yaw), torch.cos(self._odom_yaw))
        self._odom_x = self._odom_x + vx * torch.cos(self._odom_yaw) * dt
        self._odom_y = self._odom_y + vx * torch.sin(self._odom_yaw) * dt
        # Positional drift noise
        self._odom_x = self._odom_x + torch.randn(n, device=device) * self.cfg.odom_drift_std
        self._odom_y = self._odom_y + torch.randn(n, device=device) * self.cfg.odom_drift_std

        # ── Ball detection (ground-truth with FOV filter) ────
        (detected, angle_norm, dist_norm, area_norm,
         nearby_left, nearby_right, cross_net,
         nearest_dist) = self._get_ball_obs_gt(root_pos, yaw, cos_yaw, sin_yaw)

        self._last_detected = detected
        self._ball_cross_net = cross_net

        # Update ball side memory
        has_angle = detected & (angle_norm.abs() > 0.05)
        self._last_ball_side = torch.where(
            has_angle,
            torch.where(angle_norm > 0, torch.ones(n, device=device), -torch.ones(n, device=device)),
            self._last_ball_side,
        )
        self._last_ball_angle = torch.where(detected, angle_norm, self._last_ball_angle)
        # When not detected, use last known angle
        angle_norm = torch.where(detected, angle_norm, self._last_ball_angle)

        # ── Stuck signals ────────────────────────────────────
        cmd_speed = (self._last_cmd_left.abs() + self._last_cmd_right.abs()) / 200.0
        actual_speed = vx.abs()
        speed_ratio = torch.where(
            cmd_speed > 0.01,
            (actual_speed / (cmd_speed * self.cfg.max_linear_vel)).clamp(0, 1),
            torch.ones(n, device=device),
        )
        stuck_dur = self._is_stuck.float()

        # ── Odometry normalized ──────────────────────────────
        odom_x_norm = (self._odom_x / self.cfg.court_half_x).clamp(-1, 1)
        odom_y_norm = (self._odom_y / self.cfg.court_half_y).clamp(-1, 1)
        odom_yaw_norm = (self._odom_yaw / math.pi).clamp(-1, 1)

        # ── Zone counts ──────────────────────────────────────
        zone_counts = self._get_zone_counts()  # (n, 24)
        
        # Update visited zones
        cols = self.cfg.zone_grid_cols
        rows = self.cfg.zone_grid_rows
        hx = self.cfg.court_half_x
        hy = self.cfg.court_half_y
        col_idx = ((self._odom_x + hx) / (2 * hx) * cols).clamp(0, cols - 1).long()
        row_idx = ((self._odom_y + hy) / (2 * hy) * rows).clamp(0, rows - 1).long()
        cell_idx = row_idx * cols + col_idx  # (n,)
        self._zone_visited.scatter_(1, cell_idx.unsqueeze(1), True)

        zone_angle_norm, zone_dist_norm = self._get_nearest_zone_direction(
            root_pos, cos_yaw, sin_yaw,
        )
        # Cache for reward function
        self._zone_angle_norm = zone_angle_norm
        self._zone_dist_norm = zone_dist_norm

        # ── Net distance ─────────────────────────────────────
        net_dist_norm = odom_x_norm  # same as odom_x normalized

        # Total alive balls (this is a "score" the robot always knows, not vision-dependent)
        balls_frac = self._ball_alive.float().sum(dim=1) / max(self.cfg.num_balls, 1)

        # ── Assemble 44-dim obs ──────────────────────────────
        scalar_obs = torch.stack([
            detected.float(),                                          # [0]
            angle_norm,                                                # [1]
            dist_norm,                                                 # [2]
            area_norm,                                                 # [3]
            (vx / self.cfg.max_linear_vel).clamp(-1, 1),               # [4]
            (wz / self.cfg.max_angular_vel).clamp(-1, 1),              # [5]
            balls_frac.clamp(0, 1),                                    # [6]
            (cmd_speed * 2.0 - 1.0).clamp(-1, 1),                     # [7]
            (speed_ratio * 2.0 - 1.0).clamp(-1, 1),                   # [8]
            (stuck_dur * 2.0 - 1.0).clamp(-1, 1),                     # [9]
            odom_x_norm,                                               # [10]
            odom_y_norm,                                               # [11]
            odom_yaw_norm,                                             # [12]
            self._last_ball_side,                                      # [13]
            nearby_left,                                               # [14]
            nearby_right,                                              # [15]
            net_dist_norm,                                             # [16]
            cross_net,                                                 # [17]
            zone_angle_norm,                                           # [18]
            zone_dist_norm,                                            # [19]
        ], dim=-1)  # (n, 20)
        obs = torch.cat([scalar_obs, zone_counts], dim=-1)  # (n, 20 + 24 = 44)

        # Update visuals for balls (only alive ones, rest are at z=-10)
        if self.cfg.ui_window_class_type is not None or self.render_mode == "rgb_array":
            # Only visualize if not strictly headless without UI, or if rendering
            global_ball_pos = self._ball_pos + self.scene.env_origins.unsqueeze(1)
            self.ball_markers.visualize(translations=global_ball_pos.view(-1, 3))

        # ── Update observation monitor ────────────────────
        if self._obs_monitor is not None:
            self._obs_monitor.update_obs(obs)
            self._obs_monitor.update_actions(self)

        return {"policy": obs}

    # ─────────────────────────────────────────────────────────
    # Rewards
    # ─────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """Compute tensorized rewards matching PyBullet reward shaping."""
        n = self.num_envs
        device = self.device
        cfg = self.cfg

        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w

        yaw = self._quat_to_yaw(root_quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx = root_lin_vel[:, 0] * cos_yaw + root_lin_vel[:, 1] * sin_yaw
        wz = root_ang_vel[:, 2]

        reward = torch.full((n,), cfg.reward_step_penalty, device=device)
        reward_dict = {"Step Pen": reward.clone()}

        # ── Ball collection ──────────────────────────────────
        balls_collected = self._check_and_collect_balls(root_pos, yaw, cos_yaw, sin_yaw)
        ball_reward = cfg.reward_ball_collected * balls_collected.float()
        reward += ball_reward
        reward_dict["Collect"] = ball_reward

        # ── Approach reward (visible ball) ───────────────────
        nearest_dist = self._get_nearest_ball_dist(root_pos, yaw, cos_yaw, sin_yaw)
        
        both_inf = (self._prev_nearest_dist > 1e5) & (nearest_dist > 1e5)
        delta = self._prev_nearest_dist - nearest_dist
        delta = torch.where(both_inf, torch.zeros_like(delta), delta)
        
        approach_shaping = cfg.reward_approach * delta.clamp(-1.0, 1.0)
        proximity_mult = 1.0 + 0.5 / (1.0 + nearest_dist)
        vis_approach = approach_shaping * proximity_mult
        reward += vis_approach
        reward_dict["Approach (Vis)"] = vis_approach
        
        # ── Loiter / Dither Penalty ──────────────────────────
        near_any_ball = (nearest_dist < cfg.loiter_radius) & (nearest_dist < 1e5)
        self._loiter_counter = torch.where(
            near_any_ball, 
            self._loiter_counter + 1.0, 
            torch.zeros_like(self._loiter_counter)
        )
        loitering = self._loiter_counter > cfg.loiter_steps_before_penalty
        loiter_pen = torch.where(
            loitering & ~balls_collected.bool(), 
            torch.full((n,), cfg.reward_loiter_penalty, device=device), 
            torch.zeros(n, device=device)
        )
        reward += loiter_pen
        reward_dict["Loiter Pen"] = loiter_pen
        
        self._prev_nearest_dist = nearest_dist

        # ── Memory approach reward (no ball in FOV, but remembered balls exist) ──
        nearest_mem_dist = self._get_nearest_memory_dist(root_pos)
        has_memory = nearest_mem_dist < 1e5
        no_visible = ~self._last_detected
        
        mem_both_inf = (self._prev_nearest_memory_dist > 1e5) & (nearest_mem_dist > 1e5)
        mem_delta = self._prev_nearest_memory_dist - nearest_mem_dist
        mem_delta = torch.where(mem_both_inf, torch.zeros_like(mem_delta), mem_delta)
        
        mem_proximity_mult = 1.0 + 0.5 / (1.0 + nearest_mem_dist)
        mem_approach_shaping = (cfg.reward_approach * 1.5) * mem_delta.clamp(-1.0, 1.0) * mem_proximity_mult
        
        mem_active = no_visible & has_memory
        mem_approach = torch.where(mem_active, mem_approach_shaping, torch.zeros(n, device=device))
        reward += mem_approach
        reward_dict["Approach (Mem)"] = mem_approach
        self._prev_nearest_memory_dist = nearest_mem_dist

        # ── Zone-seek & Explore reward ───────────────────────
        zone_angle = self._zone_angle_norm  # (n,) in [-1,1], 0 = forward
        zone_alignment = (1.0 - 2.0 * zone_angle.abs()).clamp(min=0.0)  # 1 at front, 0 past 90°
        forward_frac = (vx / cfg.max_linear_vel).clamp(0.0, 1.0)
        not_spinning = wz.abs() < 2.0  # suppress reward while spinning in place

        zone_seek_active = no_visible & has_memory & (vx > 0.05) & not_spinning
        zone_seek_bonus = cfg.reward_zone_seek * forward_frac * zone_alignment
        zone_seek = torch.where(zone_seek_active, zone_seek_bonus, torch.zeros(n, device=device))
        reward += zone_seek
        reward_dict["Seek (Pop)"] = zone_seek

        explore_active = no_visible & (~has_memory) & (vx > 0.05) & not_spinning
        explore_bonus = cfg.reward_explore * forward_frac * zone_alignment
        explore = torch.where(explore_active, explore_bonus, torch.zeros(n, device=device))
        reward += explore
        reward_dict["Explore"] = explore

        # ── Stuck detection + penalty ────────────────────────
        self._was_stuck = self._is_stuck.clone()
        self._update_stuck_detection(vx, wz)

        stuck_p = torch.full((n,), cfg.reward_stuck_penalty, device=device)
        # Double penalty near net
        near_net = self._odom_x.abs() < cfg.net_region_x
        stuck_p = torch.where(near_net, stuck_p * 2.0, stuck_p)
        stuck_pen = torch.where(self._is_stuck, stuck_p, torch.zeros(n, device=device))
        reward += stuck_pen
        reward_dict["Stuck Pen"] = stuck_pen

        # ── Spin penalty ─────────────────────────────────────
        spinning = wz.abs() > 2.0
        spin_magnitude = wz.abs() / cfg.max_angular_vel

        # Spin when ball detected = full penalty (should drive, not spin)
        # Spin without ball = half penalty (brief search ok, prolonged spinning bad)
        spin_with_ball = spinning & self._last_detected
        spin_without_ball = spinning & (~self._last_detected)
        spin_pen = torch.where(
            spin_with_ball,
            cfg.reward_spin_penalty * spin_magnitude,
            torch.where(
                spin_without_ball,
                cfg.reward_spin_penalty * 0.5 * spin_magnitude,
                torch.zeros(n, device=device),
            ),
        )
        reward += spin_pen
        reward_dict["Spin Pen"] = spin_pen

        # ── Reverse penalty ──────────────────────────────────
        reversing = (vx < -0.05) & (~self._is_stuck)
        rev_pen = cfg.reward_reverse_penalty * (vx.abs() / cfg.max_linear_vel)
        rev_pen = torch.where(reversing, rev_pen, torch.zeros(n, device=device))
        reward += rev_pen
        reward_dict["Rev Pen"] = rev_pen

        # ── Unstuck bonus ────────────────────────────────────
        just_unstuck = self._was_stuck & (~self._is_stuck)
        unstuck_bon = torch.where(
            just_unstuck,
            torch.full((n,), cfg.reward_unstuck_bonus, device=device),
            torch.zeros(n, device=device),
        )
        reward += unstuck_bon
        reward_dict["Unstuck Bon"] = unstuck_bon

        # ── All collected bonus ──────────────────────────────
        all_done = self._ball_alive.float().sum(dim=1) == 0
        all_done_bon = torch.where(
            all_done,
            torch.full((n,), cfg.reward_all_collected_bonus, device=device),
            torch.zeros(n, device=device),
        )
        reward += all_done_bon
        reward_dict["All Done Bon"] = all_done_bon

        # ── Update reward monitor ─────────────────────────
        if self._obs_monitor is not None:
            self._obs_monitor.update_reward(
                reward,
                reward_dict,
                step=self.episode_length_buf[0].item() if len(self.episode_length_buf) > 0 else 0,
                balls_collected=self._balls_collected_total[0].item() if len(self._balls_collected_total) > 0 else 0,
            )

        return reward

    # ─────────────────────────────────────────────────────────
    # Dones & Resets
    # ─────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which envs are terminated or truncated."""
        all_collected = self._ball_alive.float().sum(dim=1) == 0
        time_out = self.episode_length_buf >= self.cfg.max_episode_steps - 1
        return all_collected, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset robot pose, ball positions, and per-env state for given envs."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n_reset = len(env_ids)
        device = self.device

        # ── Randomize robot pose ─────────────────────────────
        # Random X,Y within spawn range; random yaw
        rand_x = torch.empty(n_reset, device=device).uniform_(*self.cfg._spawn_x_range
                                                               if hasattr(self.cfg, '_spawn_x_range')
                                                               else (-10.0, 10.0))
        rand_y = torch.empty(n_reset, device=device).uniform_(-5.0, 5.0)

        # ── Avoid spawning inside the net (x=0.2, y in [-6.35, 6.35]) ──
        NET_X = 0.2
        NET_HALF_Y = 6.35
        NET_CLEARANCE = 0.5  # min distance from net center-line
        near_net = (rand_x - NET_X).abs() < NET_CLEARANCE
        near_net &= rand_y.abs() < NET_HALF_Y
        # Push robot to the same side but just outside the clearance zone
        rand_x = torch.where(near_net & (rand_x >= NET_X),
                             torch.full_like(rand_x, NET_X + NET_CLEARANCE),
                             rand_x)
        rand_x = torch.where(near_net & (rand_x < NET_X),
                             torch.full_like(rand_x, NET_X - NET_CLEARANCE),
                             rand_x)
        rand_yaw = torch.empty(n_reset, device=device).uniform_(-math.pi, math.pi)

        # Build root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0] = rand_x + self.scene.env_origins[env_ids, 0]
        root_state[:, 1] = rand_y + self.scene.env_origins[env_ids, 1]
        root_state[:, 2] = 0.22 + self.scene.env_origins[env_ids, 2]

        # Quaternion from yaw (w, x, y, z)
        root_state[:, 3] = torch.cos(rand_yaw / 2)  # w
        root_state[:, 4] = 0.0                        # x
        root_state[:, 5] = 0.0                        # y
        root_state[:, 6] = torch.sin(rand_yaw / 2)   # z
        root_state[:, 7:] = 0.0  # zero velocities

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # Zero joint velocities
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── Randomize obstacle positions (before balls so balls can avoid them) ──
        self._spawn_obstacles_for_envs(env_ids, rand_x, rand_y)

        # ── Randomize ball positions ─────────────────────────
        active_counts = self._spawn_balls_for_envs(env_ids, rand_x, rand_y, rand_yaw)
        # Mark balls alive/dead based on active count per env
        if active_counts is not None:
            idx = torch.arange(self.cfg.num_balls, device=self.device).unsqueeze(0)  # (1, nb)
            self._ball_alive[env_ids] = idx < active_counts.unsqueeze(1)
        else:
            self._ball_alive[env_ids] = True

        # ── Reset per-env state ──────────────────────────────
        self._stuck_counter[env_ids] = 0
        self._is_stuck[env_ids] = False
        self._was_stuck[env_ids] = False
        self._reverse_timer[env_ids] = 0
        self._last_cmd_left[env_ids] = 0.0
        self._last_cmd_right[env_ids] = 0.0
        self._odom_x[env_ids] = rand_x
        self._odom_y[env_ids] = rand_y
        self._odom_yaw[env_ids] = rand_yaw
        self._last_ball_side[env_ids] = 0.0
        self._last_ball_angle[env_ids] = 0.0
        self._last_detected[env_ids] = False
        self._ball_cross_net[env_ids] = 0.0
        self._spin_search_steps[env_ids] = 0.0
        self._prev_nearest_dist[env_ids] = 1e6
        self._prev_nearest_memory_dist[env_ids] = 1e6
        self._ball_discovered[env_ids] = False
        self._ball_map_pos[env_ids] = 0.0
        self._balls_collected_total[env_ids] = 0
        self._actions[env_ids] = 0.0
        self._loiter_counter[env_ids] = 0.0
        self._zone_visited[env_ids] = False
        self._focus_zone[env_ids] = -1

        # Reset obs monitor episode tracking if env 0 was reset
        if self._obs_monitor is not None:
            if isinstance(env_ids, torch.Tensor):
                if (env_ids == 0).any():
                    self._obs_monitor.reset_episode()
            elif 0 in env_ids:
                self._obs_monitor.reset_episode()

    # ─────────────────────────────────────────────────────────
    # Ball Spawning (pure tensor ops — no physics bodies needed!)
    # ─────────────────────────────────────────────────────────

    def _spawn_balls_for_envs(
        self, env_ids: torch.Tensor | Sequence[int],
        robot_x: torch.Tensor, robot_y: torch.Tensor, robot_yaw: torch.Tensor,
    ):
        """
        Spawn balls around each robot with proximity-scaled placement.

        Balls are tracked as a simple (num_envs, num_balls, 3) position tensor.
        Collection is handled via distance checks — no physics bodies for balls.
        This is much faster than spawning rigid bodies per ball per env.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        n_reset = len(env_ids)
        nb = self.cfg.num_balls
        device = self.device
        margin = 0.3
        hx = self.cfg.court_half_x - margin
        hy = self.cfg.court_half_y - margin

        if self.cfg.randomize_ball_spawn:
            # Randomize how many balls are active per env (rest hidden underground)
            active_counts = torch.randint(
                self.cfg.min_balls, self.cfg.max_balls + 1, (n_reset,), device=device
            )

            # Pick a random spawn pattern per env
            # 0=uniform, 1=clustered, 2=uniform, 3=one-side-x, 4=uniform,
            # 5=one-side-y, 6=uniform, 7=corners, 8=uniform, 9=ring
            # → 50% uniform, 50% varied
            pattern = torch.randint(0, 10, (n_reset,), device=device)

            # Start with uniform as base, then override per pattern
            ball_x = torch.empty(n_reset, nb, device=device).uniform_(-hx, hx)
            ball_y = torch.empty(n_reset, nb, device=device).uniform_(-hy, hy)

            for i in range(n_reset):
                p = pattern[i].item()
                n_active = active_counts[i].item()

                # Even numbers (0,2,4,6,8) = uniform — keep the base values as-is
                if p == 1:
                    # Clustered: pick random cluster centers, gaussian around them
                    n_clusters = torch.randint(
                        self.cfg.num_clusters_range[0],
                        self.cfg.num_clusters_range[1] + 1, (1,)
                    ).item()
                    cluster_std = torch.empty(1).uniform_(
                        self.cfg.cluster_std_range[0], self.cfg.cluster_std_range[1]
                    ).item()
                    cx = torch.empty(n_clusters, device=device).uniform_(-hx * 0.8, hx * 0.8)
                    cy = torch.empty(n_clusters, device=device).uniform_(-hy * 0.8, hy * 0.8)
                    assigns = torch.randint(0, n_clusters, (nb,), device=device)
                    ball_x[i] = cx[assigns] + torch.randn(nb, device=device) * cluster_std
                    ball_y[i] = cy[assigns] + torch.randn(nb, device=device) * cluster_std

                elif p == 3:
                    # One side X: all balls on +x or -x half
                    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                    ball_x[i] = torch.empty(nb, device=device).uniform_(0, hx) * sign
                    ball_y[i] = torch.empty(nb, device=device).uniform_(-hy, hy)

                elif p == 5:
                    # One side Y: all balls on +y or -y half
                    sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                    ball_x[i] = torch.empty(nb, device=device).uniform_(-hx, hx)
                    ball_y[i] = torch.empty(nb, device=device).uniform_(0, hy) * sign

                elif p == 7:
                    # Corners: pick 1-3 random corners, cluster balls near them
                    corners_x = torch.tensor([-hx * 0.7, -hx * 0.7, hx * 0.7, hx * 0.7], device=device)
                    corners_y = torch.tensor([-hy * 0.7, hy * 0.7, -hy * 0.7, hy * 0.7], device=device)
                    n_corners = torch.randint(1, 4, (1,)).item()
                    chosen = torch.randperm(4, device=device)[:n_corners]
                    assigns = torch.randint(0, n_corners, (nb,), device=device)
                    ball_x[i] = corners_x[chosen[assigns]] + torch.randn(nb, device=device) * 1.5
                    ball_y[i] = corners_y[chosen[assigns]] + torch.randn(nb, device=device) * 1.0

                elif p == 9:
                    # Ring: balls in a ring around court center at random radius
                    r_min, r_max = 3.0, min(hx, hy) * 0.9
                    radii = torch.empty(nb, device=device).uniform_(r_min, r_max)
                    angles = torch.empty(nb, device=device).uniform_(0, 2 * 3.14159)
                    ball_x[i] = radii * angles.cos()
                    ball_y[i] = radii * angles.sin()

                # Deactivate excess balls by moving them underground
                if n_active < nb:
                    ball_x[i, n_active:] = 0.0
                    ball_y[i, n_active:] = 0.0

            # Clamp all to court bounds
            ball_x = ball_x.clamp(-hx, hx)
            ball_y = ball_y.clamp(-hy, hy)
        else:
            # Original uniform scatter
            ball_x = torch.empty(n_reset, nb, device=device).uniform_(-hx, hx)
            ball_y = torch.empty(n_reset, nb, device=device).uniform_(-hy, hy)

        # Ensure balls aren't directly on top of the robot spawn
        # Give a 1.0m radius exclusion zone around robot
        dist_to_robot = torch.sqrt((ball_x - robot_x.unsqueeze(1))**2 + (ball_y - robot_y.unsqueeze(1))**2)
        too_close_to_robot = dist_to_robot < 1.0

        if too_close_to_robot.any():
            # direction from robot to ball
            dx_r = ball_x - robot_x.unsqueeze(1)
            dy_r = ball_y - robot_y.unsqueeze(1)
            dist_to_robot = dist_to_robot.clamp(min=1e-4) # avoid div by zero
            push_x = (dx_r / dist_to_robot) * 1.0
            push_y = (dy_r / dist_to_robot) * 1.0
            
            ball_x = torch.where(too_close_to_robot, robot_x.unsqueeze(1) + push_x, ball_x)
            ball_y = torch.where(too_close_to_robot, robot_y.unsqueeze(1) + push_y, ball_y)
            # Re-clamp
            ball_x = ball_x.clamp(-(self.cfg.court_half_x - margin), self.cfg.court_half_x - margin)
            ball_y = ball_y.clamp(-(self.cfg.court_half_y - margin), self.cfg.court_half_y - margin)

        # ── Avoid obstacle overlap ───────────────────────────
        if self.cfg.spawn_obstacles:
            obs_pos = self._obstacle_local_pos[env_ids]  # (n_reset, max_obs, 3)
            obs_active = self._obstacle_active[env_ids]   # (n_reset, max_obs)
            he = self._obstacle_sizes_t.unsqueeze(0).expand(n_reset, -1) + 0.15     # obstacle half-extent + ball margin
            he = he.unsqueeze(1) # (n_reset, 1, max_obs)

            for _ in range(3):  # up to 3 re-roll attempts
                # Distance from each ball to each obstacle: (n_reset, nb, max_obs)
                dx = ball_x.unsqueeze(2) - obs_pos[:, :, 0].unsqueeze(1)
                dy = ball_y.unsqueeze(2) - obs_pos[:, :, 1].unsqueeze(1)
                # Box overlap check (L-infinity within half-extent)
                inside = (dx.abs() < he) & (dy.abs() < he) & obs_active.unsqueeze(1)
                on_obstacle = inside.any(dim=2)  # (n_reset, nb)

                if not on_obstacle.any():
                    break

                # Re-roll overlapping balls
                new_x = torch.empty(n_reset, nb, device=device).uniform_(
                    -(self.cfg.court_half_x - margin), self.cfg.court_half_x - margin
                )
                new_y = torch.empty(n_reset, nb, device=device).uniform_(
                    -(self.cfg.court_half_y - margin), self.cfg.court_half_y - margin
                )
                ball_x = torch.where(on_obstacle, new_x, ball_x)
                ball_y = torch.where(on_obstacle, new_y, ball_y)

        ball_z = torch.full((n_reset, nb), self.cfg.ball_radius, device=device)

        # Hide inactive balls underground when using randomized counts
        if self.cfg.randomize_ball_spawn:
            idx = torch.arange(nb, device=device).unsqueeze(0)  # (1, nb)
            inactive = idx >= active_counts.unsqueeze(1)         # (n_reset, nb)
            ball_z[inactive] = -10.0

        self._ball_pos[env_ids] = torch.stack([ball_x, ball_y, ball_z], dim=-1)

        return active_counts if self.cfg.randomize_ball_spawn else None

    # ─────────────────────────────────────────────────────────
    # Obstacle Spawning
    # ─────────────────────────────────────────────────────────

    def _spawn_obstacles_for_envs(
        self, env_ids: torch.Tensor | Sequence[int],
        robot_x: torch.Tensor, robot_y: torch.Tensor,
    ):
        """
        Spawn random box obstacles on the court, avoiding the robot spawn position.

        Obstacles are kinematic rigid bodies that the robot can physically collide with.
        Unused slots are hidden below ground at z=-10.
        """
        if not self.cfg.spawn_obstacles or not self._obstacle_objects:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        n_reset = len(env_ids)
        device = self.device
        max_obs = self.cfg.max_obstacles
        margin = self.cfg.obstacle_spawn_margin

        # Random number of obstacles per env (uniform in range)
        lo, hi = self.cfg.num_obstacles_range
        n_active = torch.randint(lo, hi + 1, (n_reset,), device=device)  # (n_reset,)

        # Generate random positions for ALL slots
        obs_x = torch.empty(n_reset, max_obs, device=device).uniform_(-12.0, 12.0)
        obs_y = torch.empty(n_reset, max_obs, device=device).uniform_(-6.0, 6.0)
        
        obs_sizes = self._obstacle_sizes_t.unsqueeze(0).expand(n_reset, -1)
        obs_heights = self._obstacle_heights_t.unsqueeze(0).expand(n_reset, -1)

        def check_validity(x, y):
            # 1. Distance from robot spawn
            dist_to_robot = torch.sqrt((x - robot_x.unsqueeze(1)) ** 2 + (y - robot_y.unsqueeze(1)) ** 2)
            too_close = dist_to_robot < margin

            # 2. Net avoidance (net is at x=0.2, y in [-6.35, 6.35])
            near_net_x = (x - 0.2).abs() < (obs_sizes + 0.3)
            near_net_y = y.abs() < 6.5
            in_net = near_net_x & near_net_y

            # 3. Overlap with other obstacles
            overlap = torch.zeros(n_reset, max_obs, dtype=torch.bool, device=device)
            # Only check overlapping for indices lower than current to avoid double counting
            for i in range(max_obs):
                for j in range(i):
                    dist_x = (x[:, i] - x[:, j]).abs()
                    dist_y = (y[:, i] - y[:, j]).abs()
                    min_dist = obs_sizes[:, i] + obs_sizes[:, j] + 0.1
                    interacting = (dist_x < min_dist) & (dist_y < min_dist)
                    overlap[:, i] |= interacting
                    overlap[:, j] |= interacting

            return too_close | in_net | overlap

        invalid = check_validity(obs_x, obs_y)

        # Re-roll positions
        for _ in range(5):
            new_x = torch.empty(n_reset, max_obs, device=device).uniform_(-12.0, 12.0)
            new_y = torch.empty(n_reset, max_obs, device=device).uniform_(-6.0, 6.0)
            obs_x = torch.where(invalid, new_x, obs_x)
            obs_y = torch.where(invalid, new_y, obs_y)
            invalid = check_validity(obs_x, obs_y)

        # Deactivate obstacles still invalid after retries
        slot_idx = torch.arange(max_obs, device=device).unsqueeze(0).expand(n_reset, -1)
        active = (slot_idx < n_active.unsqueeze(1)) & (~invalid)

        # Store local positions and active mask
        self._obstacle_local_pos[env_ids] = torch.stack([obs_x, obs_y, obs_heights], dim=-1)
        self._obstacle_active[env_ids] = active

        # Write poses to each obstacle RigidObject
        for i, obs_obj in enumerate(self._obstacle_objects):
            root_state = obs_obj.data.default_root_state[env_ids].clone()
            # Active obstacles at their position; inactive at z=-10
            is_active = active[:, i]
            root_state[:, 0] = torch.where(
                is_active,
                obs_x[:, i] + self.scene.env_origins[env_ids, 0],
                self.scene.env_origins[env_ids, 0],
            )
            root_state[:, 1] = torch.where(
                is_active,
                obs_y[:, i] + self.scene.env_origins[env_ids, 1],
                self.scene.env_origins[env_ids, 1],
            )
            root_state[:, 2] = torch.where(
                is_active,
                obs_heights[:, i] + self.scene.env_origins[env_ids, 2],
                -10.0 + self.scene.env_origins[env_ids, 2],
            )
            # Identity quaternion (w=1, x=0, y=0, z=0)
            root_state[:, 3] = 1.0
            root_state[:, 4:7] = 0.0
            root_state[:, 7:] = 0.0
            obs_obj.write_root_pose_to_sim(root_state[:, :7], env_ids)

    # ─────────────────────────────────────────────────────────
    # Obstacle Occlusion (2D AABB ray intersection)
    # ─────────────────────────────────────────────────────────

    def _check_obstacle_occlusion(
        self,
        cam_x: torch.Tensor,   # (n, 1) camera x
        cam_y: torch.Tensor,   # (n, 1) camera y
        ball_x: torch.Tensor,  # (n, nb) ball x positions
        ball_y: torch.Tensor,  # (n, nb) ball y positions
    ) -> torch.Tensor:
        """
        Check if any active obstacle blocks line-of-sight from camera to each ball.

        Uses 2D slab-based AABB intersection on the line segment from camera to ball.
        Returns:
            occluded: (n, nb) bool — True if an obstacle blocks the view.
        """
        if not self.cfg.spawn_obstacles or not self._obstacle_objects:
            return torch.zeros_like(ball_x, dtype=torch.bool)

        n = cam_x.shape[0]

        # Obstacle centers: (n, max_obs)
        obs_cx = self._obstacle_local_pos[:n, :, 0]  # (n, max_obs)
        obs_cy = self._obstacle_local_pos[:n, :, 1]
        active = self._obstacle_active[:n]             # (n, max_obs)
        
        he = self._obstacle_sizes_t.unsqueeze(0).expand(n, -1).unsqueeze(1) # (n, 1, max_obs)

        # Reshape for broadcasting: camera (n,1,1), ball (n,nb,1), obstacle (n,1,max_obs)
        cx = cam_x.unsqueeze(2)           # (n, 1, 1)
        cy = cam_y.unsqueeze(2)
        bx = ball_x.unsqueeze(2)          # (n, nb, 1)
        by = ball_y.unsqueeze(2)
        ox = obs_cx.unsqueeze(1)          # (n, 1, max_obs)
        oy = obs_cy.unsqueeze(1)
        act = active.unsqueeze(1)         # (n, 1, max_obs)

        # Direction from camera to ball
        dx = bx - cx  # (n, nb, 1)
        dy = by - cy

        # AABB bounds
        xmin = ox - he  # (n, 1, max_obs)
        xmax = ox + he
        ymin = oy - he
        ymax = oy + he

        EPS = 1e-7

        # Slab intersection for X axis
        inv_dx = 1.0 / (dx + EPS * (dx >= 0).float() - EPS * (dx < 0).float())
        t1x = (xmin - cx) * inv_dx  # (n, nb, max_obs)
        t2x = (xmax - cx) * inv_dx
        t_enter_x = torch.min(t1x, t2x)
        t_exit_x = torch.max(t1x, t2x)

        # Slab intersection for Y axis
        inv_dy = 1.0 / (dy + EPS * (dy >= 0).float() - EPS * (dy < 0).float())
        t1y = (ymin - cy) * inv_dy  # (n, nb, max_obs)
        t2y = (ymax - cy) * inv_dy
        t_enter_y = torch.min(t1y, t2y)
        t_exit_y = torch.max(t1y, t2y)

        # Combined slab intersection
        t_enter = torch.max(t_enter_x, t_enter_y)
        t_exit = torch.min(t_exit_x, t_exit_y)

        # Valid intersection: slabs overlap, segment intersects (t_enter < 1, t_exit > small_eps),
        # and the obstacle must be between camera and ball (not behind camera or beyond ball)
        hit = (t_enter <= t_exit) & (t_exit > 0.01) & (t_enter < 0.99) & act

        # Occluded if ANY obstacle blocks the view: (n, nb)
        occluded = hit.any(dim=2)
        return occluded

    # ─────────────────────────────────────────────────────────
    # Ball Observation (Ground Truth with FOV Filtering)
    # ─────────────────────────────────────────────────────────

    def _get_ball_obs_gt(
        self,
        root_pos: torch.Tensor,   # (n, 3) local pos
        yaw: torch.Tensor,        # (n,)
        cos_yaw: torch.Tensor,
        sin_yaw: torch.Tensor,
    ):
        """
        Ground-truth ball observation with simulated camera FOV filtering.

        Returns:
            detected: (n,) bool
            angle_norm: (n,) float
            dist_norm: (n,) float
            area_norm: (n,) float
            nearby_left: (n,) float
            nearby_right: (n,) float
            cross_net: (n,) float
            nearest_dist: (n,) float — raw distance for approach reward
        """
        n = self.num_envs
        nb = self.cfg.num_balls
        device = self.device
        fov_half_rad = math.radians(self.cfg.cam_fov / 2.0)

        # Offset camera position forward
        cam_x = root_pos[:, 0:1] + self.cfg.camera_offset_x * cos_yaw.unsqueeze(1)
        cam_y = root_pos[:, 1:2] + self.cfg.camera_offset_x * sin_yaw.unsqueeze(1)

        # Vector from camera to each ball: (n, nb, 2)
        dx = self._ball_pos[:, :, 0] - cam_x  # (n, nb)
        dy = self._ball_pos[:, :, 1] - cam_y

        # Distance to each ball
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # (n, nb)

        # Transform to robot-local frame
        local_x = dx * cos_yaw.unsqueeze(1) + dy * sin_yaw.unsqueeze(1)   # (n, nb)
        local_y = -dx * sin_yaw.unsqueeze(1) + dy * cos_yaw.unsqueeze(1)  # (n, nb)
        angle = torch.atan2(local_y, local_x)  # (n, nb) + = left, - = right

        # Mask: in range, alive, in FOV, and not occluded by obstacles
        in_range = dist <= self.cfg.max_ball_dist_m
        in_fov = angle.abs() <= fov_half_rad
        alive = self._ball_alive
        occluded = self._check_obstacle_occlusion(
            cam_x, cam_y, self._ball_pos[:, :, 0], self._ball_pos[:, :, 1],
        )
        valid = in_range & in_fov & alive & ~occluded  # (n, nb)

        # With a focus zone, the policy still sees balls close enough to pick up on the way
        nearby = dist <= self.cfg.en_route_ball_dist_m
        visible = valid & (self._focus_mask(self._ball_pos[:, :, 0], self._ball_pos[:, :, 1]) | nearby)

        # ── Discovery: mark balls as discovered when first seen in FOV ──
        self._ball_discovered = self._ball_discovered | valid
        # Update remembered positions with distance-proportional noise (simulates vision error)
        noise_std = dist * self.cfg.ball_memory_noise_scale  # (n, nb) — farther = noisier
        noisy_x = self._ball_pos[:, :, 0] + torch.randn_like(dist) * noise_std
        noisy_y = self._ball_pos[:, :, 1] + torch.randn_like(dist) * noise_std
        self._ball_map_pos[:, :, 0] = torch.where(valid, noisy_x, self._ball_map_pos[:, :, 0])
        self._ball_map_pos[:, :, 1] = torch.where(valid, noisy_y, self._ball_map_pos[:, :, 1])

        # Net boundary check (width 12.7m -> Y in [-6.35, 6.35])
        NET_X = 0.2  # Net is at x=0.2, matching _spawn_court_colliders
        robot_x = root_pos[:, 0:1]  # (n, 1)
        robot_y = root_pos[:, 1:2]
        ball_x = self._ball_pos[:, :, 0]
        ball_y = self._ball_pos[:, :, 1]

        # Calculate Y intersection at X=NET_X (net line)
        # t is the interpolation parameter: 0 at robot, 1 at ball
        # We want the Y value where the line crosses X=NET_X
        dx_rb = ball_x - robot_x  # direction from robot to ball
        t = (NET_X - robot_x) / (dx_rb + 1e-6)
        y_int = robot_y + t * (ball_y - robot_y)

        opposite_side = ((ball_x - NET_X) * (robot_x - NET_X)) < 0
        blocked_by_net = opposite_side & (y_int.abs() <= 6.35)
        same_side = ~blocked_by_net  # Treat balls passing through the side gaps as reachable

        # For each env, find the nearest valid same-side ball; fallback to cross-side
        INF = 1e6
        dist_same = torch.where(visible & same_side, dist, torch.full_like(dist, INF))
        dist_cross = torch.where(visible & (~same_side), dist, torch.full_like(dist, INF))

        min_same, idx_same = dist_same.min(dim=1)
        min_cross, idx_cross = dist_cross.min(dim=1)

        # Prefer same-side
        use_same = min_same < INF
        use_cross = (~use_same) & (min_cross < INF)
        detected = use_same | use_cross

        best_idx = torch.where(use_same, idx_same, idx_cross)
        best_dist = torch.where(use_same, min_same, min_cross)
        cross_net_flag = torch.where(use_cross & (~use_same),
                                     torch.ones(n, device=device),
                                     torch.zeros(n, device=device))

        # Gather angle for best ball
        best_angle = torch.gather(angle, 1, best_idx.unsqueeze(1)).squeeze(1)

        # Normalize: negate so right=positive (matches YOLO convention)
        angle_norm = (-best_angle / fov_half_rad).clamp(-1, 1)
        dist_norm = (best_dist / self.cfg.max_ball_dist_m).clamp(0, 1)

        # Simulated area
        dist_mm = best_dist * 1000.0
        raw_dist_mm = dist_mm / 1.6
        focal_mm = 2.9
        sensor_w = 2.0 * focal_mm * math.tan(fov_half_rad)
        cam_w = 640.0
        cam_h = 480.0
        expected_w_px = (67.0 * focal_mm * cam_w) / (raw_dist_mm.clamp(min=10) * sensor_w)
        expected_area_px = expected_w_px ** 2
        area_norm = (expected_area_px / (cam_w * cam_h * 0.1)).clamp(0, 1)

        # Zero out obs for non-detected envs
        angle_norm = torch.where(detected, angle_norm, torch.zeros(n, device=device))
        dist_norm = torch.where(detected, dist_norm, torch.zeros(n, device=device))
        area_norm = torch.where(detected, area_norm, torch.zeros(n, device=device))
        cross_net_flag = torch.where(detected, cross_net_flag, torch.zeros(n, device=device))

        # ── Debug Draw FOV (Only for Env 0) ───────────────────
        if self.cfg.debug_draw_fov and self.fov_markers is not None and (self.cfg.ui_window_class_type is not None or self.render_mode == "rgb_array"):
            env0_cam_x = cam_x[0, 0] + self.scene.env_origins[0, 0]
            env0_cam_y = cam_y[0, 0] + self.scene.env_origins[0, 1]
            env0_cam_z = root_pos[0, 2] + self.scene.env_origins[0, 2]
            yaw0 = yaw[0]

            max_dist = self.cfg.max_ball_dist_m
            half_dist = max_dist / 2.0
            
            angles = torch.stack([yaw0 + fov_half_rad, yaw0 - fov_half_rad, yaw0])
            
            tx = env0_cam_x + half_dist * torch.cos(angles)
            ty = env0_cam_y + half_dist * torch.sin(angles)
            tz = torch.full_like(tx, env0_cam_z - 0.1) # slightly below to not block view
            
            translations = torch.stack([tx, ty, tz], dim=-1)
            
            qw = torch.cos(angles / 2.0)
            qx = torch.zeros_like(qw)
            qy = torch.zeros_like(qw)
            qz = torch.sin(angles / 2.0)

            orientations = torch.stack([qw, qx, qy, qz], dim=-1)

            self.fov_markers.visualize(
                translations=translations,
                orientations=orientations,
            )

        # ── Nearby left/right balls ──────────────────────────
        # Only balls that are visible (in range, alive, in FOV, not occluded)
        in_range_alive = valid
        SIDE_THRESH = math.radians(5.0)

        # Left balls (angle > threshold)
        left_mask = in_range_alive & (angle > SIDE_THRESH)
        dist_left = torch.where(left_mask, dist, torch.full_like(dist, INF))
        min_left_dist, min_left_idx = dist_left.min(dim=1)
        min_left_angle = torch.gather(angle, 1, min_left_idx.unsqueeze(1)).squeeze(1)
        nearby_left = torch.where(
            min_left_dist < INF,
            (-min_left_angle / fov_half_rad).clamp(-1, 0),
            torch.zeros(n, device=device),
        )

        # Right balls (angle < -threshold)
        right_mask = in_range_alive & (angle < -SIDE_THRESH)
        dist_right = torch.where(right_mask, dist, torch.full_like(dist, INF))
        min_right_dist, min_right_idx = dist_right.min(dim=1)
        min_right_angle = torch.gather(angle, 1, min_right_idx.unsqueeze(1)).squeeze(1)
        nearby_right = torch.where(
            min_right_dist < INF,
            (-min_right_angle / fov_half_rad).clamp(0, 1),
            torch.zeros(n, device=device),
        )

        # For approach reward, return the raw nearest distance (same-side preferred)
        nearest_dist = best_dist.clamp(max=INF)

        return detected, angle_norm, dist_norm, area_norm, nearby_left, nearby_right, cross_net_flag, nearest_dist

    # ─────────────────────────────────────────────────────────
    # Ball Collection
    # ─────────────────────────────────────────────────────────

    def _check_and_collect_balls(
        self,
        root_pos: torch.Tensor,
        yaw: torch.Tensor,
        cos_yaw: torch.Tensor,
        sin_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if any balls are collected (close + in front of robot).

        Returns: (num_envs,) int tensor with count of balls collected this step.
        """
        # Camera is effectively at the front bumper/hopper
        cam_x = root_pos[:, 0:1] + self.cfg.camera_offset_x * cos_yaw.unsqueeze(1)
        cam_y = root_pos[:, 1:2] + self.cfg.camera_offset_x * sin_yaw.unsqueeze(1)

        dx = self._ball_pos[:, :, 0] - cam_x
        dy = self._ball_pos[:, :, 1] - cam_y
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        # Local X relative to the front bumper (must be slightly in front or flush)
        local_x = dx * cos_yaw.unsqueeze(1) + dy * sin_yaw.unsqueeze(1)

        # Collect if within 25cm of the front bumper and roughly in front of it
        collection_radius = 0.25  
        collected = (dist < collection_radius) & (local_x > -0.05) & self._ball_alive
        balls_collected = collected.sum(dim=1)  # (n,)

        # Mark collected balls as dead and remove from discovery map
        self._ball_alive = self._ball_alive & (~collected)
        self._ball_discovered = self._ball_discovered & (~collected)
        self._ball_pos[:, :, 2] = torch.where(
            self._ball_alive, self._ball_pos[:, :, 2],
            torch.full_like(self._ball_pos[:, :, 2], -10.0),
        )
        self._balls_collected_total += balls_collected

        return balls_collected

    # ─────────────────────────────────────────────────────────
    # Stuck Detection
    # ─────────────────────────────────────────────────────────

    def _update_stuck_detection(self, vx: torch.Tensor, wz: torch.Tensor):
        """Vectorized stuck detection with minimum reverse duration."""
        actual_speed = vx.abs() + wz.abs() * 0.1
        cmd_speed = (self._last_cmd_left.abs() + self._last_cmd_right.abs()) / 200.0

        commanding = cmd_speed > self.cfg.stuck_cmd_threshold
        not_moving = actual_speed < self.cfg.stuck_speed_threshold

        # Count consecutive stuck steps
        self._stuck_counter = torch.where(
            commanding & not_moving,
            self._stuck_counter + 1,
            (self._stuck_counter - 1).clamp(min=0),
        )

        # Newly stuck: start reverse timer
        newly_stuck = (~self._is_stuck) & (self._stuck_counter >= self.cfg.stuck_steps_to_confirm)
        self._reverse_timer = torch.where(
            newly_stuck,
            torch.full_like(self._reverse_timer, float(self.cfg.stuck_min_reverse_steps)),
            self._reverse_timer,
        )

        # Tick down reverse timer
        self._reverse_timer = (self._reverse_timer - 1).clamp(min=0)

        # Stuck while reverse timer is active
        self._is_stuck = self._reverse_timer > 0

    # ─────────────────────────────────────────────────────────
    # Zone-Based Ball Memory
    # ─────────────────────────────────────────────────────────

    def set_focus_zone(self, zone: int, env_ids: torch.Tensor | None = None) -> None:
        """Steer the policy to a flat zone index (row * cols + col); -1 clears the focus."""
        if env_ids is None:
            self._focus_zone[:] = zone
        else:
            self._focus_zone[env_ids] = zone

    def _focus_mask(self, bx: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        """(n, nb) True for balls the policy may act on: all of them, or those in the focus zone."""
        cols = self.cfg.zone_grid_cols
        rows = self.cfg.zone_grid_rows
        hx = self.cfg.court_half_x
        hy = self.cfg.court_half_y
        col_idx = ((bx + hx) / (2 * hx) * cols).clamp(0, cols - 1).long()
        row_idx = ((by + hy) / (2 * hy) * rows).clamp(0, rows - 1).long()
        focus = self._focus_zone.unsqueeze(1)
        return (focus < 0) | (row_idx * cols + col_idx == focus)

    def _get_zone_counts(self) -> torch.Tensor:
        """
        Count *discovered* alive balls in a grid over the court.

        Only balls the robot has seen (in FOV) are included — uses the
        remembered positions from _ball_map_pos, not the true positions.

        Grid: zone_grid_cols x zone_grid_rows (default 6x4 = 24 cells).
        Returns: (n, cols*rows) normalized fractions.
        """
        cols = self.cfg.zone_grid_cols
        rows = self.cfg.zone_grid_rows
        hx = self.cfg.court_half_x
        hy = self.cfg.court_half_y

        bx = self._ball_map_pos[:, :, 0]  # (n, nb) — remembered positions
        by = self._ball_map_pos[:, :, 1]
        known = (self._ball_discovered & self._ball_alive & self._focus_mask(bx, by)).float()  # (n, nb)

        # Map ball positions to grid cell indices
        # X: [-hx, hx] → [0, cols-1], Y: [-hy, hy] → [0, rows-1]
        col_idx = ((bx + hx) / (2 * hx) * cols).clamp(0, cols - 1).long()  # (n, nb)
        row_idx = ((by + hy) / (2 * hy) * rows).clamp(0, rows - 1).long()  # (n, nb)
        cell_idx = row_idx * cols + col_idx  # (n, nb) — flat index

        # Scatter-add known balls into grid cells
        n = known.shape[0]
        num_cells = cols * rows
        grid = torch.zeros(n, num_cells, device=known.device)
        grid.scatter_add_(1, cell_idx, known)

        total = known.sum(dim=1, keepdim=True).clamp(min=1)
        return grid / total

    def _get_nearest_zone_direction(
        self,
        root_pos: torch.Tensor,   # (n, 3) local pos
        cos_yaw: torch.Tensor,    # (n,)
        sin_yaw: torch.Tensor,    # (n,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the nearest grid cell that contains discovered balls and return
        its direction in robot-local frame.

        Returns:
            zone_angle_norm: (n,) angle to nearest populated cell, normalized to [-1, 1]
                             (1 = directly behind, 0 = forward, +/- = left/right via sign)
            zone_dist_norm:  (n,) distance to nearest populated cell, normalized to [0, 1]
                             Returns 0 for both if no populated cells exist.
        """
        n = self.num_envs
        cols = self.cfg.zone_grid_cols
        rows = self.cfg.zone_grid_rows
        hx = self.cfg.court_half_x
        hy = self.cfg.court_half_y
        num_cells = cols * rows
        device = self.device

        # Cell-occupancy from discovered+alive balls
        bx = self._ball_map_pos[:, :, 0]
        by = self._ball_map_pos[:, :, 1]
        known = (self._ball_discovered & self._ball_alive & self._focus_mask(bx, by)).float()  # (n, nb)
        col_idx = ((bx + hx) / (2 * hx) * cols).clamp(0, cols - 1).long()
        row_idx = ((by + hy) / (2 * hy) * rows).clamp(0, rows - 1).long()
        cell_idx = row_idx * cols + col_idx
        grid = torch.zeros(n, num_cells, device=device)
        grid.scatter_add_(1, cell_idx, known)
        populated = grid > 0  # (n, num_cells)

        # Cell centroids in world (local-to-env) coordinates
        cell_w = (2 * hx) / cols
        cell_h = (2 * hy) / rows
        c_arange = torch.arange(cols, device=device).float()
        r_arange = torch.arange(rows, device=device).float()
        cell_cx = -hx + (c_arange + 0.5) * cell_w  # (cols,)
        cell_cy = -hy + (r_arange + 0.5) * cell_h  # (rows,)
        # Build (num_cells,) flat centroid arrays
        cy_grid, cx_grid = torch.meshgrid(cell_cy, cell_cx, indexing="ij")
        flat_cx = cx_grid.reshape(-1)  # (num_cells,)
        flat_cy = cy_grid.reshape(-1)

        # Vector from robot to each cell centroid: (n, num_cells)
        dx = flat_cx.unsqueeze(0) - root_pos[:, 0:1]  # (n, num_cells)
        dy = flat_cy.unsqueeze(0) - root_pos[:, 1:2]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)

        # Mask: consider populated cells FIRST
        INF = 1e6
        dist_pop = torch.where(populated, dist, torch.full_like(dist, INF))
        nearest_pop_dist, nearest_pop_idx = dist_pop.min(dim=1)  # (n,)
        has_pop = nearest_pop_dist < INF

        # Mask: consider unvisited cells SECOND
        dist_unvis = torch.where((~populated) & (~self._zone_visited), dist, torch.full_like(dist, INF))
        nearest_unvis_dist, nearest_unvis_idx = dist_unvis.min(dim=1)
        has_unvis = nearest_unvis_dist < INF

        # Target selection: populated overrides unvisited
        nearest_idx = torch.where(has_pop, nearest_pop_idx, nearest_unvis_idx)
        has_any = has_pop | has_unvis

        # A mission focus overrides the policy's own choice of zone
        focused = self._focus_zone >= 0
        nearest_idx = torch.where(focused, self._focus_zone, nearest_idx)
        has_any = has_any | focused

        # Gather absolute (x,y) of the target cell
        target_cx = torch.gather(flat_cx.unsqueeze(0).expand(n, -1), 1, nearest_idx.unsqueeze(1)).squeeze(1)
        target_cy = torch.gather(flat_cy.unsqueeze(0).expand(n, -1), 1, nearest_idx.unsqueeze(1)).squeeze(1)

        # ── Net-aware waypointing ────────────────────────────
        # Net is at x=NET_X and spans y∈[-6.35, 6.35]; the gaps are at y > +6.35 and y < -6.35.
        # If the target cell is on the OTHER side of the net, route the bot to the nearest
        # gap waypoint first, then once it's past the net, point to the cell directly.
        NET_X = 0.2
        gap_y = self.cfg.net_gap_y  # e.g. 6.5 — just past the net edge
        robot_x = root_pos[:, 0]
        robot_y = root_pos[:, 1]
        cross_net = ((target_cx - NET_X) * (robot_x - NET_X)) < 0  # opposite sides of net
        # Choose nearest gap based on robot's current Y
        gap_y_signed = torch.where(robot_y >= 0, torch.full_like(robot_y, gap_y),
                                                 torch.full_like(robot_y, -gap_y))
        # If still on the robot's side of the net, aim at the gap; otherwise aim at the cell
        wp_x = torch.where(cross_net, torch.full_like(target_cx, NET_X), target_cx)
        wp_y = torch.where(cross_net, gap_y_signed, target_cy)

        # Vector from robot to waypoint
        best_dx = wp_x - robot_x
        best_dy = wp_y - robot_y
        nearest_dist = torch.sqrt(best_dx * best_dx + best_dy * best_dy + 1e-8)

        # Transform into robot-local frame
        local_x = best_dx * cos_yaw + best_dy * sin_yaw
        local_y = -best_dx * sin_yaw + best_dy * cos_yaw
        angle = torch.atan2(local_y, local_x)  # (-pi, pi); 0 = forward

        # Normalize: angle / pi → [-1, 1] (sign: + = left, - = right; ±1 = behind)
        angle_norm = (angle / math.pi).clamp(-1, 1)
        # Distance normalized by court diagonal
        max_dist = math.sqrt((2 * hx) ** 2 + (2 * hy) ** 2)
        dist_norm = (nearest_dist / max_dist).clamp(0, 1)

        # Zero out if no populated cells
        angle_norm = torch.where(has_any, angle_norm, torch.zeros_like(angle_norm))
        dist_norm = torch.where(has_any, dist_norm, torch.zeros_like(dist_norm))

        return angle_norm, dist_norm

    # ─────────────────────────────────────────────────────────
    # Nearest Ball Distance (for approach reward)
    # ─────────────────────────────────────────────────────────

    def _get_nearest_ball_dist(
        self,
        root_pos: torch.Tensor,
        yaw: torch.Tensor,
        cos_yaw: torch.Tensor,
        sin_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Get distance to nearest alive ball in FOV (same-side preferred)."""
        fov_half_rad = math.radians(self.cfg.cam_fov / 2.0)
        INF = 1e6

        dx = self._ball_pos[:, :, 0] - root_pos[:, 0:1]
        dy = self._ball_pos[:, :, 1] - root_pos[:, 1:2]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        local_x = dx * cos_yaw.unsqueeze(1) + dy * sin_yaw.unsqueeze(1)
        local_y = -dx * sin_yaw.unsqueeze(1) + dy * cos_yaw.unsqueeze(1)
        angle = torch.atan2(local_y, local_x)

        in_range = dist <= self.cfg.max_ball_dist_m
        in_fov = angle.abs() <= fov_half_rad
        # Camera position for occlusion check
        cam_x = root_pos[:, 0:1] + self.cfg.camera_offset_x * cos_yaw.unsqueeze(1)
        cam_y = root_pos[:, 1:2] + self.cfg.camera_offset_x * sin_yaw.unsqueeze(1)
        occluded = self._check_obstacle_occlusion(
            cam_x, cam_y, self._ball_pos[:, :, 0], self._ball_pos[:, :, 1],
        )
        valid = in_range & in_fov & self._ball_alive & ~occluded

        NET_X = 0.2  # Net is at x=0.2, matching _spawn_court_colliders
        robot_x = root_pos[:, 0:1]
        robot_y = root_pos[:, 1:2]
        ball_x = self._ball_pos[:, :, 0]
        ball_y = self._ball_pos[:, :, 1]

        dx_rb = ball_x - robot_x
        t = (NET_X - robot_x) / (dx_rb + 1e-6)
        y_int = robot_y + t * (ball_y - robot_y)

        opposite_side = ((ball_x - NET_X) * (robot_x - NET_X)) < 0
        blocked_by_net = opposite_side & (y_int.abs() <= 6.35)
        same_side = ~blocked_by_net

        dist_same = torch.where(valid & same_side, dist, torch.full_like(dist, INF))
        dist_cross = torch.where(valid & (~same_side), dist, torch.full_like(dist, INF))

        min_same = dist_same.min(dim=1).values
        min_cross = dist_cross.min(dim=1).values

        use_same = min_same < INF
        return torch.where(use_same, min_same, min_cross)

    # ─────────────────────────────────────────────────────────
    # Nearest Memory Distance (for memory approach reward)
    # ─────────────────────────────────────────────────────────

    def _get_nearest_memory_dist(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Distance to nearest discovered-but-not-currently-visible ball.

        Uses remembered positions from _ball_map_pos. Same-side preferred.
        Returns INF (1e6) if no discovered balls exist.
        """
        INF = 1e6
        known_alive = self._ball_discovered & self._ball_alive  # (n, nb)

        dx = self._ball_map_pos[:, :, 0] - root_pos[:, 0:1]
        dy = self._ball_map_pos[:, :, 1] - root_pos[:, 1:2]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        # Net blocking (same logic as _get_nearest_ball_dist)
        NET_X = 0.2
        robot_x = root_pos[:, 0:1]
        robot_y = root_pos[:, 1:2]
        ball_x = self._ball_map_pos[:, :, 0]
        ball_y = self._ball_map_pos[:, :, 1]

        dx_rb = ball_x - robot_x
        t = (NET_X - robot_x) / (dx_rb + 1e-6)
        y_int = robot_y + t * (ball_y - robot_y)

        opposite_side = ((ball_x - NET_X) * (robot_x - NET_X)) < 0
        blocked_by_net = opposite_side & (y_int.abs() <= 6.35)
        same_side = ~blocked_by_net

        dist_same = torch.where(known_alive & same_side, dist, torch.full_like(dist, INF))
        dist_cross = torch.where(known_alive & (~same_side), dist, torch.full_like(dist, INF))

        min_same = dist_same.min(dim=1).values
        min_cross = dist_cross.min(dim=1).values

        use_same = min_same < INF
        return torch.where(use_same, min_same, min_cross)

    # ─────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
        """Extract yaw angle from quaternion (w, x, y, z) format."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

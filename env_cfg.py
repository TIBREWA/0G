"""
TennisbotEnvCfg — Isaac Lab DirectRLEnv configuration for the tennis-ball-collecting rover.

This mirrors rl/config.py but adapted to Isaac Lab's dataclass-based configuration system.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

# ── Paths ────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_ROVER_URDF = str(Path(__file__).resolve().parent / "model/rover/cylinder.urdf")

# ── Court Dimensions (must match world.urdf) ─────────────────
FENCE_HALF_X = 15.0        # m — half-length
FENCE_HALF_Y = 7.5         # m — half-width

# ── Spawn Ranges ─────────────────────────────────────────────
SPAWN_X_RANGE = (-10.0, 10.0)
SPAWN_Y_RANGE = (-5.0, 5.0)


@configclass
class TennisbotEnvCfg(DirectRLEnvCfg):
    """Configuration for the Tennisbot Isaac Lab environment."""

    # ── Simulation ───────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,          # render every 2 physics steps
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9,
            dynamic_friction=0.8,
            friction_combine_mode="multiply",  # was "max" — "max" ignores per-body friction
            restitution=0.0,
        ),
    )
    decimation: int = 5             # env step = 5 physics steps → ~24 Hz control

    # ── Scene ────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=24576,
        env_spacing=30.0,          # Increased massively to prevent overlap on long axis 
    )

    # ── Robot ────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=_ROVER_URDF,
            fix_base=False,
            replace_cylinders_with_capsules=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="velocity",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=10.0,
                    damping=0.0,
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.22),
        ),
        actuators={
            "front_wheels": ImplicitActuatorCfg(
                joint_names_expr=["front_.*_wheel_joint"],
                effort_limit_sim=1.0,       # 9.8 kg·cm ≈ 0.96 N·m per motor
                velocity_limit_sim=31.4,    # 300 RPM
                stiffness=0.0,              # pure velocity control
                damping=5.0,                # velocity tracking gain
            ),
            "rear_caster": ImplicitActuatorCfg(
                joint_names_expr=["rear_wheel_joint"],
                effort_limit_sim=0.0,       # free-spinning, no motor (matches PyBullet force=0)
                velocity_limit_sim=100.0,   # no limit
                stiffness=0.0,
                damping=0.0,                # zero resistance — free caster
            ),
        },
    )

    # ── Observation / Action Dims ────────────────────────────
    observation_space: int = 44
    action_space: int = 2

    # ── Episode ──────────────────────────────────────────────
    episode_length_s: float = 10000 * (5 / 120.0)  # 10000 steps × decimation/sim_dt ≈ 416s
    max_episode_steps: int = 10000

    # ── Balls ────────────────────────────────────────────────
    num_balls: int = 30
    ball_radius: float = 0.0335 # m
    randomize_ball_spawn: bool = True
    min_balls: int = 10                    # when randomizing, spawn between min and max
    max_balls: int = 30
    num_clusters_range: tuple = (1, 5)     # how many cluster centers
    cluster_std_range: tuple = (0.5, 3.0)  # std dev of each cluster in meters

    # ── Obstacles ────────────────────────────────────────────
    spawn_obstacles: bool = True
    num_obstacles_range: tuple = (3, 5)
    max_obstacles: int = 5              # fixed slots per env (unused ones hidden below ground)
    obstacle_width_range: tuple = (0.4, 0.8)   # (min, max) half-extent
    obstacle_height_range: tuple = (0.2, 0.4)  # (min, max) half-height
    obstacle_spawn_margin: float = 1.5  # min distance from robot spawn

    # ── Wheel Physics ────────────────────────────────────────
    wheel_radius: float = 0.05     # m
    track_width: float = 0.27      # m  (±0.135 Y offset in cylinder.urdf)
    max_throttle: float = 0.6      # Maximum throttle the AI is allowed to use
    max_wheel_rad_s: float = 31.4 * 0.6  # rad/s — 60% of 300 RPM real motor

    # ── Camera FOV (for ground-truth "simulated vision") ─────
    light_scale: float = 1.0       # brightness only; has no effect on observations
    cam_fov: float = 78.0          # degrees
    en_route_ball_dist_m: float = 4.0  # balls this close stay visible outside a focus zone
    camera_offset_x: float = 0.35  # m — distance forward from robot center
    debug_draw_fov: bool = True   # Set to False to disable the cyan/yellow FOV visualization rays

    # ── Normalization ────────────────────────────────────────
    max_ball_dist_m: float = 6.0
    max_linear_vel: float = (31.4 * 0.6) * 0.05    # max_wheel_rad_s × wheel radius
    max_angular_vel: float = 5.0
    court_half_x: float = FENCE_HALF_X
    court_half_y: float = FENCE_HALF_Y
    zone_grid_cols: int = 6             # X divisions (along court length)
    zone_grid_rows: int = 4             # Y divisions (along court width)
    net_region_x: float = 0.3       # narrowed from 0.8 to be closer to net (~nose touch)

    # ── Reward Shaping ───────────────────────────────────────
    # Keep max ratio under ~50:1. RunningStandardScaler normalizes magnitudes,
    # so only ratios between rewards matter — not absolute values.
    reward_ball_collected: float = 5.0
    reward_approach: float = 0.3
    reward_step_penalty: float = -0.02
    reward_all_collected_bonus: float = 10.0
    reward_stuck_penalty: float = -0.3
    reward_unstuck_bonus: float = 1.0
    reward_spin_penalty: float = -0.05
    reward_reverse_penalty: float = 0
    reward_cross_net_navigate: float = 0.1    # reward for moving toward net gap when balls are cross-net
    reward_zone_seek: float = 0.08            # bonus for driving forward toward nearest populated zone
    reward_explore: float = 0.05              # bonus for driving forward toward nearest unvisited zone
    net_gap_y: float = 6.5                     # Y coord of net gap waypoint (just past net edge at 6.35)
    spin_search_max_steps: int = 60            # after this many spin steps, search bonus decays to zero

    # ── Loiter / Dither Penalty ──────────────────────────────
    reward_loiter_penalty: float = -0.15
    loiter_radius: float = 1.0
    loiter_steps_before_penalty: int = 30
    
    # ── Sensor Noise (sim-to-real gap) ────────────────────────
    odom_drift_std: float = 0#0.0002         # positional drift noise per step (m)
    odom_yaw_drift_std: float = 0#0.00006    # yaw drift noise per step (rad), ~5°/min
    ball_memory_noise_scale: float =0 #0.03  # position noise = scale * distance (m per m)

    # ── Stuck Detection ──────────────────────────────────────
    stuck_speed_threshold: float = 0.05
    stuck_cmd_threshold: float = 0.05
    stuck_steps_to_confirm: int = 15
    stuck_min_reverse_steps: int = 50    # once stuck, stay stuck for at least this many steps to force escape
    stuck_max_duration_norm: float = 100.0

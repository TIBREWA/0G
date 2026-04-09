"""
train.py — Train the tennis-bot RL agent with skrl PPO on Isaac Lab.

GPU-accelerated replacement for rl/train.py (PyBullet + SB3).

Usage
-----
    # Train with 4096 parallel envs (headless, maximum speed)
    python isaac/train.py --num_envs 4096 --headless

    # Train with GUI visualization (fewer envs for visibility)
    python isaac/train.py --num_envs 16

    # Resume from checkpoint
    python isaac/train.py --resume isaac/runs/latest/checkpoints/best_agent.pt

    # Custom config
    python isaac/train.py --num_envs 2048 --timesteps 5000000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in path so we can import 'isaac'
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # <-- MUST BE IMPORTED BEFORE OMNIVERSE TO PREVENT TLS BLOCK ERROR

# ── Must initialize SimulationApp BEFORE any other Isaac imports ──
from isaaclab.app import AppLauncher

# Parse args that AppLauncher needs (--headless, --num_envs, etc.)
parser = argparse.ArgumentParser(description="Train Tennisbot with skrl PPO on Isaac Lab")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training timesteps")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--play", action="store_true", default=False, help="Run evaluation (no training)")
parser.add_argument("--remote", action="store_true", default=False, help="Manual keyboard control (arrow keys)")
parser.add_argument("--show_monitor", action="store_true", default=False, help="Show the AI brain HUD in remote mode")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--experiment_name", type=str, default="tennisbot_ppo", help="Experiment name for logging")

# AppLauncher adds --headless, --device, etc.
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab and skrl ────────────────────
import yaml

from isaaclab.envs import DirectRLEnv

from isaac.env_cfg import TennisbotEnvCfg
from isaac.tennisbot_env import TennisbotEnv

# skrl imports
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# ── Project paths ────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_ISAAC_DIR = str(Path(__file__).resolve().parent)
_CFG_PATH = os.path.join(_ISAAC_DIR, "agents", "ppo_cfg.yaml")


# ── Neural Network Models ───────────────────────────────────

class PolicyNetwork(GaussianMixin, Model):
    """Gaussian policy network for continuous actions."""

    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20.0, max_log_std=2.0,
                 reduction="sum", initial_log_std=-1.0, hidden_layers=None):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        hidden = hidden_layers or [256, 256]
        layers = []
        in_dim = self.num_observations
        for h in hidden:
            lin = torch.nn.Linear(in_dim, h)
            torch.nn.init.orthogonal_(lin.weight, gain=1.414)
            torch.nn.init.constant_(lin.bias, 0.0)
            layers.extend([
                lin,
                torch.nn.LayerNorm(h),
                torch.nn.SiLU(),
            ])
            in_dim = h
        last_lin = torch.nn.Linear(in_dim, self.num_actions)
        torch.nn.init.orthogonal_(last_lin.weight, gain=0.01)
        torch.nn.init.constant_(last_lin.bias, 0.0)
        layers.append(last_lin)

        self.net = torch.nn.Sequential(*layers)
        self.log_std_parameter = torch.nn.Parameter(torch.full((self.num_actions,), initial_log_std))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class ValueNetwork(DeterministicMixin, Model):
    """Deterministic value network (critic)."""

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, hidden_layers=None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        hidden = hidden_layers or [512, 512, 256]
        layers = []
        in_dim = self.num_observations
        for h in hidden:
            lin = torch.nn.Linear(in_dim, h)
            torch.nn.init.orthogonal_(lin.weight, gain=1.414)
            torch.nn.init.constant_(lin.bias, 0.0)
            layers.extend([
                lin,
                torch.nn.LayerNorm(h),
                torch.nn.SiLU(),
            ])
            in_dim = h
        last_lin = torch.nn.Linear(in_dim, 1)
        torch.nn.init.orthogonal_(last_lin.weight, gain=1.0)
        torch.nn.init.constant_(last_lin.bias, 0.0)
        layers.append(last_lin)

        self.net = torch.nn.Sequential(*layers)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def train(args):
    """Main training function."""
    if args.seed < 0:
        import random
        args.seed = random.randint(0, 2**31 - 1)
        print(f"[INFO] Using random seed: {args.seed}")
    set_seed(args.seed)

    # ── Load YAML config ─────────────────────────────────────
    with open(_CFG_PATH, "r") as f:
        yaml_cfg = yaml.safe_load(f)

    # ── Create environment ───────────────────────────────────
    env_cfg = TennisbotEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = TennisbotEnv(cfg=env_cfg)

    # Wrap for skrl
    from skrl.envs.wrappers.torch import wrap_env
    env = wrap_env(env, wrapper="isaaclab")

    device = env.device

    # ── Create models ────────────────────────────────────────
    hidden = yaml_cfg.get("policy", {}).get("hidden_layers", [256, 256])
    initial_log_std = yaml_cfg.get("policy", {}).get("initial_log_std", -1.0)

    models = {
        "policy": PolicyNetwork(
            env.observation_space, env.action_space, device,
            initial_log_std=initial_log_std, hidden_layers=hidden,
        ),
        "value": ValueNetwork(
            env.observation_space, env.action_space, device,
            hidden_layers=yaml_cfg.get("value", {}).get("hidden_layers", [256, 256]),
        ),
    }

    # ── Create memory ────────────────────────────────────────
    rollouts = yaml_cfg.get("rollouts", 32)
    memory = RandomMemory(memory_size=rollouts, num_envs=args.num_envs, device=device)

    # ── Configure PPO agent ──────────────────────────────────
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "rollouts": rollouts,
        "learning_epochs": yaml_cfg.get("learning_epochs", 10),
        "mini_batches": yaml_cfg.get("mini_batches", 8),
        "discount_factor": yaml_cfg.get("discount_factor", 0.99),
        "lambda": yaml_cfg.get("lambda_factor", 0.95),
        "learning_rate": yaml_cfg.get("learning_rate", 1e-4),
        "learning_rate_scheduler": CosineAnnealingLR,
        "learning_rate_scheduler_kwargs": {
            "T_max": max(1, int(args.timesteps / (args.num_envs * rollouts))),
            "eta_min": 1e-5, # Ensures it doesn't decay purely to 0
        },
        "random_timesteps": yaml_cfg.get("random_timesteps", 0),
        "learning_starts": yaml_cfg.get("learning_starts", 0),
        "grad_norm_clip": yaml_cfg.get("grad_norm_clip", 0.5),
        "ratio_clip": yaml_cfg.get("ratio_clip", 0.2),
        "value_clip": yaml_cfg.get("value_clip", 0.2),
        "clip_predicted_values": yaml_cfg.get("clip_predicted_values", True),
        "entropy_loss_scale": yaml_cfg.get("entropy_loss_scale", 0.0),
        "value_loss_scale": yaml_cfg.get("value_loss_scale", 0.5),
        "kl_threshold": yaml_cfg.get("kl_threshold", 0.0),
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "experiment": {
            "directory": os.path.join(_ISAAC_DIR, "runs"),
            "experiment_name": args.experiment_name,
            "write_interval": 1,        # Log every single update (first one in ~500k steps)
            "checkpoint_interval": 1000,
        },
    })

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # ── Resume from checkpoint ───────────────────────────────
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[TRAIN] Resuming from {args.resume}")
            agent.load(args.resume)
        else:
            print(f"[ERROR] Checkpoint not found: {args.resume}")
            print(f"  CWD: {os.getcwd()}")
            simulation_app.close()
            sys.exit(1)

    # ── Create trainer and start ─────────────────────────────
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={
            "timesteps": args.timesteps,
            "headless": args.headless if hasattr(args, "headless") else True,
        },
    )

    print(f"\n{'=' * 60}")
    print(f"  Training PPO with skrl on Isaac Lab")
    print(f"  Timesteps:   {args.timesteps:,}")
    print(f"  Num Envs:    {args.num_envs}")
    print(f"  Rollouts:    {rollouts}")
    print(f"  Mini-batches: {agent_cfg['mini_batches']}")
    print(f"  LR:          {agent_cfg['learning_rate']}")
    print(f"  Device:      {device}")
    print(f"  Experiment:  {args.experiment_name}")
    print(f"  Play Mode:   {args.play}")
    print(f"{'=' * 60}\n")

    if args.play:
        print("[TRAIN] Proceeding to inference (evaluation) mode...")
        # Evaluation
        trainer.eval()
    else:
        print("[TRAIN] Starting training...")
        trainer.train()

    # ── Cleanup ──────────────────────────────────────────────
    env.close()
    simulation_app.close()


def remote_control(args):
    """Run a single environment with manual keyboard control (arrow keys) + omni.ui HUD."""
    import carb.input
    import omni.appwindow
    import omni.ui as ui
    import math
    import time
    import collections
    from isaac.env_cfg import TennisbotEnvCfg
    from isaac.tennisbot_env import TennisbotEnv

    # ── Constants ────────────────────────────────────────────
    MAX_WHEEL_VEL = 31.4
    WHEEL_RADIUS = 0.05
    MAX_LINEAR_SPEED = MAX_WHEEL_VEL * WHEEL_RADIUS
    SPEED_PRESETS = [0.2, 0.4, 0.6, 0.8, 1.0]
    HUD_UPDATE_EVERY = 8

    # ── Create environment (1 env) ───────────────────────────
    env_cfg = TennisbotEnvCfg()
    env_cfg.scene.num_envs = 1
    env = TennisbotEnv(cfg=env_cfg)
    device = env.device

    # ── Disable the obs_monitor (AI brain HUD) ───────────────
    if not args.show_monitor:
        env._obs_monitor = None

    # ── Build omni.ui HUD window ─────────────────────────────
    hud_window = ui.Window("Rover Telemetry", width=340, height=480)
    hud_labels = {}
    with hud_window.frame:
        with ui.VStack(spacing=4):
            # Title
            ui.Label("ROVER  TELEMETRY", style={"font_size": 18, "color": 0xFFFFFFFF},
                     height=30, alignment=ui.Alignment.CENTER)
            ui.Separator(height=2)

            # FPS
            with ui.HStack(height=22):
                ui.Label("FPS", width=100, style={"color": 0xFF78DC50})
                hud_labels["fps"] = ui.Label("0.0", style={"color": 0xFF78DC50})

            ui.Separator(height=2)

            # Speed
            with ui.HStack(height=22):
                ui.Label("LINEAR SPEED", width=140, style={"color": 0xFFA0A0A0})
                hud_labels["speed"] = ui.Label("0.000 m/s", style={"color": 0xFF00DCDC})
            with ui.HStack(height=16):
                hud_labels["speed_bar"] = ui.ProgressBar(style={"color": 0xFF00DCDC})
                hud_labels["speed_bar"].model.set_value(0.0)

            ui.Spacer(height=4)

            # Throttle
            with ui.HStack(height=22):
                ui.Label("THROTTLE", width=140, style={"color": 0xFFA0A0A0})
                hud_labels["throttle"] = ui.Label("100%", style={"color": 0xFFF09620})
            with ui.HStack(height=16):
                hud_labels["throttle_bar"] = ui.ProgressBar(style={"color": 0xFFF09620})
                hud_labels["throttle_bar"].model.set_value(1.0)

            ui.Separator(height=2)

            # Wheel commands
            ui.Label("WHEEL CMD (normalized)", style={"color": 0xFFA0A0A0}, height=20)
            with ui.HStack(height=22):
                ui.Label("L", width=20, style={"color": 0xFF3CD23C})
                hud_labels["vl"] = ui.Label("+0.00", style={"color": 0xFF3CD23C})
            with ui.HStack(height=16):
                hud_labels["vl_bar"] = ui.ProgressBar(style={"color": 0xFF3CD23C})
                hud_labels["vl_bar"].model.set_value(0.0)
            with ui.HStack(height=22):
                ui.Label("R", width=20, style={"color": 0xFFD23C3C})
                hud_labels["vr"] = ui.Label("+0.00", style={"color": 0xFFD23C3C})
            with ui.HStack(height=16):
                hud_labels["vr_bar"] = ui.ProgressBar(style={"color": 0xFFD23C3C})
                hud_labels["vr_bar"].model.set_value(0.0)

            ui.Separator(height=2)

            # Pose
            ui.Label("POSE", style={"color": 0xFFA0A0A0}, height=20)
            hud_labels["pos"] = ui.Label("x=  0.00 m   y=  0.00 m",
                                         style={"color": 0xFFFFFFFF}, height=22)
            hud_labels["yaw"] = ui.Label("yaw=    0.0 deg",
                                         style={"color": 0xFFFFFFFF}, height=22)

            ui.Separator(height=2)

            # Balls
            ui.Label("BALLS", style={"color": 0xFFA0A0A0}, height=20)
            hud_labels["balls"] = ui.Label("Alive: 0   Collected: 0",
                                           style={"color": 0xFFC8B428}, height=22)

            ui.Separator(height=2)

            # Key hints
            ui.Label("Up/Dn fwd/back   Up+Lr arc turn",
                     style={"color": 0xFF787878, "font_size": 12}, height=16)
            ui.Label("Lr alone spin   1-5 speed %",
                     style={"color": 0xFF787878, "font_size": 12}, height=16)
            ui.Label("Space stop   R reset",
                     style={"color": 0xFF787878, "font_size": 12}, height=16)

    def update_hud(d):
        hud_labels["fps"].text = f"{d['fps']:.1f}"
        hud_labels["speed"].text = f"{d['speed']:.3f} m/s"
        hud_labels["speed_bar"].model.set_value(min(1.0, abs(d['speed']) / MAX_LINEAR_SPEED))
        hud_labels["throttle"].text = f"{int(d['throttle'] * 100)}%"
        hud_labels["throttle_bar"].model.set_value(d['throttle'])
        hud_labels["vl"].text = f"{d['vl']:+.2f}"
        hud_labels["vl_bar"].model.set_value(min(1.0, abs(d['vl'])))
        hud_labels["vr"].text = f"{d['vr']:+.2f}"
        hud_labels["vr_bar"].model.set_value(min(1.0, abs(d['vr'])))
        hud_labels["pos"].text = f"x={d['x']:6.2f} m   y={d['y']:6.2f} m"
        hud_labels["yaw"].text = f"yaw={d['yaw']:7.1f} deg"
        hud_labels["balls"].text = f"Alive: {d['balls_alive']}   Collected: {d['collected']}"

    # ── Keyboard setup ───────────────────────────────────────
    input_iface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()

    key_states = {"up": False, "down": False, "left": False, "right": False,
                  "space": False, "r": False}
    throttle_trigger = [None]

    def _on_key_event(event, *_a, **_kw):
        pressed = event.type in (carb.input.KeyboardEventType.KEY_PRESS,
                                 carb.input.KeyboardEventType.KEY_REPEAT)
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE

        mapping = {
            carb.input.KeyboardInput.UP: "up",
            carb.input.KeyboardInput.DOWN: "down",
            carb.input.KeyboardInput.LEFT: "left",
            carb.input.KeyboardInput.RIGHT: "right",
            carb.input.KeyboardInput.W: "up",
            carb.input.KeyboardInput.S: "down",
            carb.input.KeyboardInput.A: "left",
            carb.input.KeyboardInput.D: "right",
            carb.input.KeyboardInput.SPACE: "space",
            carb.input.KeyboardInput.R: "r",
        }
        key = mapping.get(event.input)
        if key is not None:
            if pressed:
                key_states[key] = True
            elif released:
                key_states[key] = False

        # Throttle presets (1-5) on trigger only
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            preset_map = {
                carb.input.KeyboardInput.KEY_1: 0,
                carb.input.KeyboardInput.KEY_2: 1,
                carb.input.KeyboardInput.KEY_3: 2,
                carb.input.KeyboardInput.KEY_4: 3,
                carb.input.KeyboardInput.KEY_5: 4,
            }
            idx = preset_map.get(event.input)
            if idx is not None:
                throttle_trigger[0] = SPEED_PRESETS[idx]
        return True

    sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_key_event)

    print("\n" + "=" * 58)
    print("  REMOTE CONTROL — Isaac Sim")
    print("=" * 58)
    print("  Up/Dn  fwd/back   Up+Arrow  arc turn")
    print("  Arrow alone  spin in place   1-5  speed preset")
    print("  Space  stop   R  reset   Close window to exit")
    print("=" * 58 + "\n")

    # ── State ────────────────────────────────────────────────
    throttle = 0.6
    vl_cmd = vr_cmd = 0.0
    step = 0
    fps_buf = collections.deque(maxlen=60)
    t_last = time.perf_counter()
    fps = 0.0

    obs, _ = env.reset()

    while simulation_app.is_running():
        # Check throttle update
        if throttle_trigger[0] is not None:
            throttle = throttle_trigger[0]
            throttle_trigger[0] = None

        # ── Diff-drive input logic (matching differential1.py) ──
        target_vl = 0.0
        target_vr = 0.0

        if key_states["space"]:
            target_vl = target_vr = 0.0
        else:
            up    = key_states["up"]
            down  = key_states["down"]
            left  = key_states["left"]
            right = key_states["right"]

            fwd = throttle
            arc = fwd * 0.40

            if up and left:
                target_vl, target_vr = arc, fwd
            elif up and right:
                target_vl, target_vr = fwd, arc
            elif down and left:
                target_vl, target_vr = -arc, -fwd
            elif down and right:
                target_vl, target_vr = -fwd, -arc
            elif up:
                target_vl, target_vr = fwd, fwd
            elif down:
                target_vl, target_vr = -fwd, -fwd
            elif left:
                target_vl, target_vr = -fwd, fwd
            elif right:
                target_vl, target_vr = fwd, -fwd
            else:
                target_vl, target_vr = 0.0, 0.0

        # Gentle acceleration / deceleration (exponential smoothing)
        alpha = 0.3  # smoothing factor (0.0 to 1.0, lower = smoother but more lag)
        vl_cmd = vl_cmd + alpha * (target_vl - vl_cmd)
        vr_cmd = vr_cmd + alpha * (target_vr - vr_cmd)

        # Reset
        if key_states["r"]:
            key_states["r"] = False
            obs, _ = env.reset()
            step = 0
            print("  [RESET]")
            continue

        vl_cmd = max(-1.0, min(1.0, vl_cmd))
        vr_cmd = max(-1.0, min(1.0, vr_cmd))
        action = torch.tensor([[vl_cmd, vr_cmd]], device=device)

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        # FPS
        t_now = time.perf_counter()
        dt = t_now - t_last
        t_last = t_now
        fps_buf.append(1.0 / dt if dt > 0 else 0.0)
        fps = sum(fps_buf) / len(fps_buf)

        # HUD update (~30Hz)
        if step % HUD_UPDATE_EVERY == 0:
            root_pos = env.robot.data.root_pos_w[0] - env.scene.env_origins[0]
            root_quat = env.robot.data.root_quat_w[0]
            root_lin_vel = env.robot.data.root_lin_vel_w[0]
            yaw_rad = 2.0 * torch.atan2(root_quat[3], root_quat[0]).item()
            cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
            speed = root_lin_vel[0].item() * cos_y + root_lin_vel[1].item() * sin_y

            update_hud(dict(
                fps=fps, speed=abs(speed), throttle=throttle,
                vl=vl_cmd, vr=vr_cmd,
                x=root_pos[0].item(), y=root_pos[1].item(),
                yaw=math.degrees(yaw_rad),
                balls_alive=env._ball_alive.sum().item(),
                collected=env._balls_collected_total[0].item(),
            ))

        if terminated.any() or truncated.any():
            print(f"  Episode ended at step {step}. Resetting...")
            obs, _ = env.reset()
            step = 0

    input_iface.unsubscribe_to_keyboard_events(keyboard, sub)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    if args.remote:
        remote_control(args)
    else:
        train(args)

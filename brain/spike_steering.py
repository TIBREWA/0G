"""Does the focus-zone hook steer a trained policy? Compares steered envs against a control group.

    python brain/spike_steering.py --headless --checkpoints runs/best_agent.pt runs/agent_74000.pt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

import torch  # must be imported before Omniverse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from isaac.brain.policy import LegsPolicy
from isaac.env_cfg import TennisbotEnvCfg
from isaac.tennisbot_env import TennisbotEnv

cfg = TennisbotEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.debug_draw_fov = False
env = TennisbotEnv(cfg=cfg)
cols, rows = cfg.zone_grid_cols, cfg.zone_grid_rows


def zone_of(xy: torch.Tensor) -> torch.Tensor:
    col = ((xy[:, 0] + cfg.court_half_x) / (2 * cfg.court_half_x) * cols).clamp(0, cols - 1).long()
    row = ((xy[:, 1] + cfg.court_half_y) / (2 * cfg.court_half_y) * rows).clamp(0, rows - 1).long()
    return row * cols + col


def far_targets(xy: torch.Tensor) -> torch.Tensor:
    """A random zone at least two columns from each robot, so a chance visit is unlikely."""
    spawn_col = zone_of(xy) % cols
    scores = torch.rand(xy.shape[0], cols * rows, device=xy.device)
    zone_cols = torch.arange(cols * rows, device=xy.device) % cols
    scores[(zone_cols.unsqueeze(0) - spawn_col.unsqueeze(1)).abs() < 2] = -1.0
    return scores.argmax(dim=1)


def run(policy: LegsPolicy, steer: bool) -> tuple[float, float, float]:
    """Return the share of envs reaching target, median steps to reach it, and mean balls collected."""
    n = env.num_envs
    obs, _ = env.reset()
    env.episode_length_buf[:] = 0
    target = far_targets((env.robot.data.root_pos_w - env.scene.env_origins)[:, :2])
    if steer:
        env._focus_zone[:] = target
        obs = env._get_observations()
    reached_at = torch.full((n,), -1, device=env.device)
    for step in range(args.steps):
        obs, _, _, _, _ = env.step(policy.act(obs["policy"]))
        xy = (env.robot.data.root_pos_w - env.scene.env_origins)[:, :2]
        hit = (zone_of(xy) == target) & (reached_at < 0)
        reached_at[hit] = step
        if steer:
            env._focus_zone[:] = target
    done = reached_at >= 0
    median = reached_at[done].float().median().item() if done.any() else float("nan")
    return done.float().mean().item(), median, env._balls_collected_total.float().mean().item()


def ball_zones() -> torch.Tensor:
    """Flat zone index of every ball, (n, nb)."""
    return zone_of(env._ball_pos[:, :, :2].reshape(-1, 2)).reshape(env.num_envs, -1)


def pick_densest() -> torch.Tensor:
    """Test stand-in for the 0G brain: the zone holding the most uncollected balls."""
    counts = torch.zeros(env.num_envs, cols * rows, device=env.device)
    counts.scatter_add_(1, ball_zones(), env._ball_alive.float())
    return counts.argmax(dim=1)


def run_mission(policy: LegsPolicy, steer: bool) -> tuple[float, float]:
    """Return the mean share of balls collected and the share taken from the assigned zone."""
    obs, _ = env.reset()
    env.episode_length_buf[:] = 0
    total = env._ball_alive.sum(dim=1).clamp(min=1).float()
    assigned = pick_densest()
    in_zone = torch.zeros(env.num_envs, device=env.device)
    collected = torch.zeros(env.num_envs, device=env.device)
    finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if steer:
        env._focus_zone[:] = assigned
        obs = env._get_observations()
    for _ in range(args.steps):
        before = env._ball_alive.clone()
        zones = ball_zones()
        obs, _, terminated, truncated, _ = env.step(policy.act(obs["policy"]))
        # A terminated env collected everything and was reset, so read its result from `before`.
        taken = torch.where(terminated.unsqueeze(1), before, before & ~env._ball_alive)
        count = ~finished
        collected[count] += taken[count].sum(dim=1).float()
        in_zone[count] += (taken & (zones == assigned.unsqueeze(1)))[count].sum(dim=1).float()
        finished |= terminated | truncated
        cleared = ~(env._ball_alive & (ball_zones() == assigned.unsqueeze(1))).any(dim=1)
        assigned = torch.where(cleared, pick_densest(), assigned)
        if steer:
            env._focus_zone[:] = assigned
    return (collected / total).mean().item(), (in_zone.sum() / collected.sum().clamp(min=1)).item()


for path in args.checkpoints:
    policy = LegsPolicy(path, device=env.device)
    for label, steer in (("alone", False), ("mission", True)):
        cleared, obedience = run_mission(policy, steer)
        print(f"MISSION {path} {label}: collected={cleared:.0%} from_assigned_zone={obedience:.0%}", flush=True)
    for label, steer in (("control", False), ("steered", True)):
        share, median, balls = run(policy, steer)
        print(f"RESULT {Path(path).name} {label}: reached={share:.0%} median_steps={median:.0f} balls={balls:.1f}", flush=True)

env.close()
simulation_app.close()

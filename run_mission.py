"""Run one rover mission with 0G making the decisions at every checkpoint.

    python isaac/run_mission.py --headless --brief "A match starts soon, clear the near side."

Reads ZG_API_KEY and ZG_MODELS (comma separated) from the shell or isaac/.env. If 0G cannot be reached the rover stops and
waits for it; there is no local decision path.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from isaac.brain.settings import load_env

load_env()
MODELS = [m.strip() for m in os.environ.get("ZG_MODELS", "").split(",") if m.strip()]
API_KEY = os.environ.get("ZG_API_KEY")
if __name__ == "__main__" and not (API_KEY and MODELS):
    print(f"Set ZG_API_KEY and ZG_MODELS in {ROOT / '.env'}", file=sys.stderr)
    raise SystemExit(1)

import torch  # must be imported before Omniverse
from PIL import Image

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--brief", default="A match starts in 3 minutes. Clear the near side first. Only cross the net if there is time.")
parser.add_argument("--limit_s", type=float, default=180.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--checkpoint", default=str(ROOT / "runs/agent_44000.pt"))
parser.add_argument("--near_side_first", action="store_true", help="Only offer far-side zones once the starting side has nothing left")
parser.add_argument("--frames", default=None, help="Save overhead camera frames to this folder")
parser.add_argument("--out", default=str(ROOT / "episodes"))
parser.add_argument("--no_pacing", action="store_true", help="Run faster than real time")
parser.add_argument("--sever_at_s", type=float, default=None, help="Cut the route to 0G at this mission time")
parser.add_argument("--restore_at_s", type=float, default=None, help="Restore the route to 0G")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.enable_cameras or args.frames is not None
simulation_app = AppLauncher(args).app

from isaac.brain.client import ROUTER_URL, ZeroGBrain
from isaac.brain.episode_log import EpisodeLog
from isaac.brain.mission import STEPS_PER_SECOND, Mission, Phase, Snapshot
from isaac.brain.policy import LegsPolicy
from isaac.env_cfg import TennisbotEnvCfg
from isaac.tennisbot_env import TennisbotEnv

UNREACHABLE_URL = "http://127.0.0.1:9/v1"
POSE_EVERY_STEPS = 6
FRAME_EVERY_STEPS = 2


def snapshot(env: TennisbotEnv, step: int, visited: list[bool]) -> Snapshot:
    """The rover's own view: its localised pose and the balls it has seen, never undiscovered ones.

    Pose comes from the simulator, standing in for a localisation system, because the
    dead-reckoned odometry drifts by several metres within a mission.
    """
    cfg = env.cfg
    cols, rows = cfg.zone_grid_cols, cfg.zone_grid_rows
    pose = env.robot.data.root_pos_w[0] - env.scene.env_origins[0]
    x, y = pose[0].item(), pose[1].item()
    col = int((x + cfg.court_half_x) / (2 * cfg.court_half_x) * cols)
    row = int((y + cfg.court_half_y) / (2 * cfg.court_half_y) * rows)
    zone = min(max(row, 0), rows - 1) * cols + min(max(col, 0), cols - 1)
    visited[zone] = True
    known = [0] * (cols * rows)
    for ball in (env._ball_discovered[0] & env._ball_alive[0]).nonzero().flatten().tolist():
        bx, by = env._ball_map_pos[0, ball].tolist()
        b_col = min(max(int((bx + cfg.court_half_x) / (2 * cfg.court_half_x) * cols), 0), cols - 1)
        b_row = min(max(int((by + cfg.court_half_y) / (2 * cfg.court_half_y) * rows), 0), rows - 1)
        known[b_row * cols + b_col] += 1
    return Snapshot(
        step=step, x=x, y=y, yaw_deg=math.degrees(env._quat_to_yaw(env.robot.data.root_quat_w)[0].item()), zone=zone,
        stuck=bool(env._is_stuck[0].item()), collected=int(env._balls_collected_total[0].item()),
        known_balls=known, visited=list(visited),
    )


def main() -> None:
    cfg = TennisbotEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.debug_draw_fov = False
    frames = Path(args.frames) if args.frames else None
    if frames:
        frames.mkdir(parents=True, exist_ok=True)
        cfg.viewer.eye = (0.0, -10.0, 31.0)
        cfg.viewer.lookat = (0.0, -0.8, 0.0)
        cfg.light_scale = 0.12
        cfg.viewer.resolution = (1280, 720)
    env = TennisbotEnv(cfg=cfg, render_mode="rgb_array" if frames else None)
    policy = LegsPolicy(args.checkpoint, device=env.device)
    obs, _ = env.reset()

    alive = env._ball_alive[0].clone()
    log = EpisodeLog(f"ep-{int(time.time())}-s{args.seed}", {
        "brief": args.brief, "seed": args.seed, "limit_s": args.limit_s,
        "near_side_first": args.near_side_first,
        "steps_per_second": STEPS_PER_SECOND,
        "policy_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "router": ROUTER_URL, "models": MODELS,
        "court": {"half_x": cfg.court_half_x, "half_y": cfg.court_half_y,
                  "cols": cfg.zone_grid_cols, "rows": cfg.zone_grid_rows},
        "balls": [[i, *[round(v, 3) for v in env._ball_pos[0, i, :2].tolist()]]
                  for i in alive.nonzero().flatten().tolist()],
        "obstacles": [[round(v, 3) for v in env._obstacle_local_pos[0, i, :2].tolist()]
                      for i in env._obstacle_active[0].nonzero().flatten().tolist()],
    })
    brain = ZeroGBrain(API_KEY, MODELS)
    limit_steps = int(args.limit_s * STEPS_PER_SECOND)
    mission = Mission(brain, log, args.brief, limit_steps, near_side_first=args.near_side_first)
    zero_action = torch.zeros(1, 2, device=env.device)

    started = time.monotonic()
    step, last_phase = 0, None
    visited = [False] * (cfg.zone_grid_cols * cfg.zone_grid_rows)
    while simulation_app.is_running():
        now_s = step / STEPS_PER_SECOND
        if args.sever_at_s is not None and args.sever_at_s <= now_s:
            brain.base_url = UNREACHABLE_URL
        if args.restore_at_s is not None and args.restore_at_s <= now_s:
            brain.base_url = ROUTER_URL.rstrip("/")

        command = mission.update(snapshot(env, step, visited))
        if command.phase is not last_phase:
            print(f"[{now_s:6.1f}s] {command.phase.value} zone={mission.focus_zone} {mission.last_reason}", flush=True)
            last_phase = command.phase
        if command.phase is Phase.DONE:
            break

        env.set_focus_zone(command.focus_zone)
        action = policy.act(obs["policy"]) if command.drive else zero_action
        before = env._ball_alive[0].clone()
        obs, _, terminated, truncated, _ = env.step(action)
        for ball in (before & ~env._ball_alive[0]).nonzero().flatten().tolist():
            log.ball_collected(step, ball)
        if step % POSE_EVERY_STEPS == 0:
            pose = env.robot.data.root_pos_w[0] - env.scene.env_origins[0]
            log.pose(step, pose[0].item(), pose[1].item(), env._odom_yaw[0].item(), command.phase.value, command.focus_zone)
        if frames and step % FRAME_EVERY_STEPS == 0:
            Image.fromarray(env.render()).save(frames / f"{step:06d}.jpg", quality=88)
        step += 1
        if terminated.any():
            mission.end_reason = "all_balls_collected"
            break
        if truncated.any():
            break
        if not args.no_pacing:
            time.sleep(max(0.0, started + step / STEPS_PER_SECOND - time.monotonic()))

    mission.close()
    log.finish(step, mission.end_reason)
    print("Saved", log.save(Path(args.out)), log.data["result"], flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

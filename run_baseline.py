"""Policy alone, with no 0G and no focus zone, on the same course and time limit as a mission.

    python isaac/run_baseline.py --headless --seed 8
"""
import json, sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import torch
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--seed", type=int); p.add_argument("--limit_s", type=float, default=180.0)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); app = AppLauncher(args).app
from isaac.brain.policy import LegsPolicy
from isaac.env_cfg import TennisbotEnvCfg
from isaac.tennisbot_env import TennisbotEnv
cfg = TennisbotEnvCfg(); cfg.scene.num_envs = 1; cfg.seed = args.seed; cfg.debug_draw_fov = False
env = TennisbotEnv(cfg=cfg); policy = LegsPolicy(str(ROOT / "runs/agent_44000.pt"), device=env.device); obs, _ = env.reset()
alive0 = env._ball_alive[0].clone(); ids = alive0.nonzero().flatten().tolist()
balls = [[i, *[round(v, 3) for v in env._ball_pos[0, i, :2].tolist()]] for i in ids]
order = []
start_x = (env.robot.data.root_pos_w[0] - env.scene.env_origins[0])[0].item()
times, end, why = [], int(args.limit_s * 24), "time_limit"
for step in range(int(args.limit_s * 24)):
    before = env._ball_alive[0].clone()
    obs, _, term, trunc, _ = env.step(policy.act(obs["policy"]))
    gone = (before & ~env._ball_alive[0]).nonzero().flatten().tolist()
    times += [step] * len(gone)
    order += [[step, ball] for ball in gone]
    if term.any() or trunc.any():
        end, why = step + 1, "terminated" if term.any() else "truncated"; break
print("RESULT", json.dumps({"seed": args.seed, "balls": balls, "order": order, "start_x": start_x, "total": len(ids), "collected": len(times), "end_s": round(end/24,1), "why": why, "last_s": round(times[-1]/24) if times else None, "t50": round(times[len(ids)//2-1]/24) if len(times)>=len(ids)//2 else None}), flush=True)
env.close(); app.close()

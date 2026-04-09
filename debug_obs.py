import torch
import math
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# Initialize AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaac.env_cfg import TennisbotEnvCfg
from isaac.tennisbot_env import TennisbotEnv

def debug_observations():
    cfg = TennisbotEnvCfg()
    cfg.scene.num_envs = 1
    # Force at least 10 balls to ensure something is in FOV eventually
    cfg.num_balls = 10
    cfg.debug_draw_fov = False
    env = TennisbotEnv(cfg)
    
    print("\n" + "="*60)
    print("  OBSERVATION DEBUG: AI-BRAIN SIGNALS")
    print("="*60)
    
    # Run for 50 steps with no input (robot stays still)
    any_detected = False
    for i in range(50):
        # Action [0, 0] = Still
        action = torch.tensor([[0.0, 0.0]], device=env.device)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"][0]
        
        detected = bool(obs[0].item())
        angle = obs[1].item()
        dist = obs[2].item()
        area = obs[3].item()
        vx_obs = obs[4].item()
        
        if detected:
            any_detected = True

        if i % 5 == 0:
            status = "🎾 DETECTED" if detected else "⚪ NOTHING"
            print(f"Step {i:2d} | {status} | Angle: {angle:+.3f} | Dist: {dist:.3f} | Area: {area:.3f} | VX_Obs: {vx_obs:.3f}")
            
    print("\n" + "="*60)
    if not any_detected:
        print("❌ CRITICAL: The AI is BLIND (Detected=0 consistently).")
        print("   This explains why it is reversing: it's searching but cannot find any balls.")
        print("   Possible reason: The FOV or coordinate system is flipped.")
    else:
        print("✅ SUCCESS: The AI can see balls!")
        print("   If it is still reversing, then the Brain is just poorly trained.")
    print("="*60 + "\n")
        
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    debug_observations()

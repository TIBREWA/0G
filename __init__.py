"""
Isaac Lab Tennisbot Environment Package.

Registers the TennisbotIsaac-v0 environment with Gymnasium.
"""

import gymnasium

gymnasium.register(
    id="TennisbotIsaac-v0",
    entry_point="isaaclab.envs:DirectRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaac.env_cfg:TennisbotEnvCfg",
        "task_entry_point": "isaac.tennisbot_env:TennisbotEnv",
    },
    disable_env_checker=True,
)

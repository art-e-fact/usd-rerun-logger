import gymnasium as gym
import rerun as rr
import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher()

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (  # noqa: E402
    FrankaReachEnvCfg,
)

from usd_rerun_logger.env_wrapper import LogRerun  # noqa: E402

env = gym.make("Isaac-Reach-Franka-v0", cfg=FrankaReachEnvCfg())
rr.init("franka_example", spawn=True)
env = LogRerun(env)
env.reset()
for _ in range(100):
    action_np = env.action_space.sample()
    action = torch.as_tensor(action_np)
    env.step(action)

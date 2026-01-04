# Rerun.io logger for USD and NVIDIA Omniverse apps

### :construction: Development preview. Work in progress.

## Usage examples:

### Logging plain USD scene
```py
rr.init("orange_example", spawn=True)
stage = Usd.Stage.Open("robot.usd"))
logger = UsdRerunLogger(stage)
logger.log_stage()
```


### Logging Isaac Sim scene:
```py
world = World()

rr.init()
logger = UsdRerunLogger(world.stage, path_filter=["!*BlackGrid*"])

while app_running:
    world.step()
    rr.set_time(timeline="sim", duration=sim_time)
    logger.log_stage()
```

### Logging Isaac Lab environment:
```py
rr.init()
logger = IsaacLabRerunLogger(env.scene)
while looping:
    env.step(action)
    rr.set_time(
        timeline="sim",
        duration=env.common_step_counter * env.step_dt,
    )
    logger.log_scene()
```


### Logging Gymnasium environment:
```py
env = gym.make("Isaac-Reach-Franka-v0", cfg=FrankaReachEnvCfg())
rr.init("franka_example", spawn=True)
env = LogRerun(env)
env.reset()
for _ in range(100):
    action_np = env.action_space.sample()
    action = torch.as_tensor(action_np)
    env.step(action)
```

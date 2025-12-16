## Rerun logger for USD and NVIDIA Omniverse apps

# :construction: Development preview. Work in progress.

Usage notes:

With Isaac Sim:
```py
world = World()

rr.init()
logger = UsdRerunLogger(world.stage, path_filter=["!*BlackGrid*"])

while app_running:
    world.step()
    rr.set_time(timeline="sim", duration=sim_time)
    logger.log_stage()
```

With Isaac Lab:
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
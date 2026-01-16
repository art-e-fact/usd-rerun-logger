# Rerun.io logger for USD and NVIDIA Omniverse apps

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://art-e-fact.github.io/usd-rerun-logger/)
[![PyPI](https://img.shields.io/pypi/v/usd-rerun-logger)](https://pypi.org/project/usd-rerun-logger/)

### :construction: This is a development preview. Expect breaking changes before 1.0.0 :construction: 

## Contents
 - [Install](#install)
 - [Log generic USD Scenes](#log-generic-usd-scenes)
 - [Log Isaac Sim scenes](#log-isaac-sim-scenes)
 - [Log Isaac Lab InteractiveScenes](#log-isaac-lab-interactivescenes)
 - [Wrap Gymnasium environments](#log-gymnasium-environments)

## Install

 1. You can install [usd-rerun-logger](https://pypi.org/project/usd-rerun-logger/)  from PyPI:
```
pip install usd-rerun-logger
```
 2. Install OpenUSD (`pxr`). This is a **user-managed dependency** to avoid version conflicts (e.g. with `isaac-sim` which bundles its own version).
    - If you use an Omniverse app like **Isaac Sim** or **Isaac Lab**: You can skip this step.
    - Otherwise, install the standard library from PyPI:
      ```bash
      pip install usd-core
      ```


## Log generic USD Scenes

This utility can traverse any standard USD scene and log transforms, geometries and color textures to [Rerun.io](https://rerun.io/).

[API reference](https://art-e-fact.github.io/usd-rerun-logger/generated/usd_rerun_logger.UsdRerunLogger.html#usd_rerun_logger.UsdRerunLogger)

[Executable examples](./examples/README.md#usdrerunlogger-examples)

This is the lowest level logger. See IsaacLabRerunLogger or the LogRerun wrapper for higher level APIs

To log a simple scene:
```py
from pxr import Usd
from usd_rerun_logger import UsdRerunLogger

# Init Rerun.io
rr.init("orange_example", spawn=True)
# Create the USD Stage
stage = Usd.Stage.Open("orange.usd")
# Log the stage
logger = UsdRerunLogger(stage)
logger.log_stage()
```

## Log Isaac Sim scenes
```py
from usd_rerun_logger import UsdRerunLogger

rr.init("IsaacSim", spawn=True)
logger = UsdRerunLogger(world.stage, path_filter=["!*BlackGrid*"])

while app_running:
    world.step()
    # Set the simulation time on the timeline (see the full examples)
    rr.set_time(timeline="sim", duration=sim_time)
    logger.log_stage()
```

## Log Isaac Lab InteractiveScenes

To make training faster, Isaac Lab doesn't update the USD Stage and the latest transforms are only available through the InteractiveScene API. The `IsaacLabRerunLogger` will merge the 3D objects and initial poses from the USD stage with the transforms parsed from the InteractiveScene.

[API reference](https://art-e-fact.github.io/usd-rerun-logger/generated/usd_rerun_logger.IsaacLabRerunLogger.html)

```py
from usd_rerun_logger import IsaacLabRerunLogger

rr.init("IsaacLab", spawn=True)
logger = IsaacLabRerunLogger(env.scene)
while looping:
    env.step(action)
    rr.set_time(timeline="sim_steps", sequence=env.common_step_counter)
    logger.log_scene()
```

## Log Gymnasium environments

`LogRerun` is a drop-in replacement for Gymnasium's [RecordVideo wrapper](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo). Using Rerun.io instead of recording videos has multiple benefits, like no need to worry about camera angles, and file size doesn't increase linearly with the length of the run.

[API reference](https://art-e-fact.github.io/usd-rerun-logger/generated/usd_rerun_logger.LogRerun.html)

[Executable examples](./examples/README.md#isaac-lab-gymnasium-wrapper-examples)

```py
from usd_rerun_logger import LogRerun

env = gym.make("Isaac-Reach-Franka-v0", cfg=FrankaReachEnvCfg())
rr.init("franka_example", spawn=True)
env = LogRerun(env)
env.reset()
for _ in range(100):
    action_np = env.action_space.sample()
    action = torch.as_tensor(action_np)
    env.step(action)
```


### Known limitations
 - Not all shaders are supported. In OpenUSD, color textures can be represented in a  wide variety of ways, many of them specific to certain implementations. We try to support as many shader formats as possible. Please open an issue when you have a .usd* file that we fail to parse properly.

Also, see the [issues](https://github.com/art-e-fact/usd-rerun-logger/issues) for features we're still working on.

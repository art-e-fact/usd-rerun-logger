## UsdRerunLogger examples

### Prerequisites

Install the OpenUSD package (either via `pip install usd-core` or installing `isaacsim` or `isaaclab`)

### Logging USD files

Use [`log_usd.py`](./log_usd.py) to visualize any USD file with Rerun.io.

```bash
# List available preset USD files
uv run examples/log_usd.py
# Log a preset
uv run examples/log_usd.py orange
uv run examples/log_usd.py so101

# Log any USD file
uv run examples/log_usd.py /path/to/your/scene.usda
```

#### Minimal code

The core logic is just:

```python
from pxr import Usd
from usd_rerun_logger import UsdRerunLogger
import rerun as rr

rr.init("my_scene", spawn=True)
stage = Usd.Stage.Open("scene.usda")
logger = UsdRerunLogger(stage)
logger.log_stage()
```

## Isaac Sim examples

TODO

## Isaac Lab examples

TODO


## Isaac Lab Gymnasium wrapper examples

### Prequisites

Install Isaac Lab following the [official instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) or by running:

```bash
uv pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
```

### Logging random steps in Isaac Lab environments

```bash
# List available environment presets
uv run examples/isaac_lab_env.py

# Run a specific environment
uv run examples/isaac_lab_env.py h1
uv run examples/isaac_lab_env.py go2
uv run examples/isaac_lab_env.py franka-reach

# Run with custom number of steps
uv run examples/isaac_lab_env.py digit --steps 200
```

#### Minimal code

```python
import gymnasium as gym
import rerun as rr
import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher()

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY
from usd_rerun_logger.env_wrapper import LogRerun

rr.init("my_env", spawn=True)
env = gym.make("Isaac-Velocity-Rough-H1-Play-v0", cfg=H1FlatEnvCfg_PLAY())
env = LogRerun(env)
env.reset()

for _ in range(100):
    action = torch.as_tensor(env.action_space.sample())
    env.step(action)
```


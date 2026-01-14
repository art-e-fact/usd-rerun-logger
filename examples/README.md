## UsdRerunLogger examples

Use [`usd_example.py`](./usd_example.py) to visualize any USD file with Rerun.io.

```bash
# List available preset USD files
uv run examples/usd_example.py

# Log a preset
uv run examples/usd_example.py orange
uv run examples/usd_example.py so101

# Log any USD file
uv run examples/usd_example.py /path/to/your/scene.usda
```

### Minimal code

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

## Isaac Lab wrapper example

See [`isaac_lab_wrapper_franka.py`](./isaac_lab_wrapper_franka.py) for an example using the `LogRerun` Gymnasium wrapper with Isaac Lab.
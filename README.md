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
```

## Publishing to Pypi

After merging to `main`:

1. checkout `main`, then:
    * update the `unreleased` section of the CHANGELOG to reflect the new version you are going to release.
    * update pyproject.yaml with new version number or run `uv version x.x.x` 
    * usual `git add`, `git commit`
    * `git tag  -a x.x.x -m "x.x.x"` 
    * `git push` then `git push --tags`

2. Build and publish to pypi (you will need a pypi token)
```
# Build
uv build
# Publish
uv publish --token <pypi token>
```

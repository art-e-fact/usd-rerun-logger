import itertools

import numpy as np
import rerun as rr

from .util import assert_isaac_lab_dependency, assert_usd_core_dependency

assert_usd_core_dependency()
assert_isaac_lab_dependency()


from isaaclab.scene import InteractiveScene  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from .transfom import log_usd_transform  # noqa: E402
from .visual import log_visuals  # noqa: E402

# Note: In Isaac Lab, we can't read poses directly from the USD: https://github.com/isaac-sim/IsaacLab/issues/3472#issuecomment-3299713710


class IsaacLabRerunLogger:
    def __init__(
        self,
        scene: "InteractiveScene",
        logged_envs: int | list[int] = 0,
    ):
        self._scene = scene
        self._prev_transforms: dict[
            str, np.ndarray
        ] = {}  # Track the last logged poses (position + orientation)
        self._prev_usd_transforms: dict[
            str, Gf.Matrix4d
        ] = {}  # Save last USD transforms to know the scale
        self._scene_structure_logged = False
        self._logged_envs = (
            [logged_envs] if isinstance(logged_envs, int) else logged_envs
        )

    def log_scene(self):
        if self._scene is None or self._scene.stage is None:
            return

        assets = itertools.chain(
            self._scene.articulations.items(),
            self._scene.rigid_objects.items(),
        )
        for obj_name, obj in assets:
            poses = obj.data.body_pose_w.cpu().numpy()  # shape: (num_bodies, 3)

            for env_id in range(self._scene.num_envs):
                # Skip logging for unlisted environments
                if env_id not in self._logged_envs:
                    continue

                root_path = obj.cfg.prim_path.replace(".*", str(env_id))

                for body_index, body_name in enumerate(obj.body_names):
                    # TODO: Find out what is a robust way to find the prim path of a body
                    if body_name != obj_name:
                        body_path = f"{root_path}/{body_name}"
                    else:
                        body_path = root_path

                    # Log the meshes once
                    if not self._scene_structure_logged:
                        self._log_usd_subtree(body_path)

                    pose = poses[env_id][body_index]

                    # Skip logging if the transform hasn't changed
                    if body_path in self._prev_transforms and np.array_equal(
                        self._prev_transforms[body_path], pose
                    ):
                        continue

                    self._prev_transforms[body_path] = pose

                    if body_path in self._prev_usd_transforms:
                        usd_transform = self._prev_usd_transforms[body_path]
                        # Extract scale from USD transform
                        scale = Gf.Transform(usd_transform).GetScale()
                    else:
                        scale = None
                    rr.log(
                        body_path,
                        rr.Transform3D(
                            translation=pose[:3],
                            quaternion=pose[[4, 5, 6, 3]],
                            scale=scale,
                        ),
                    )

        # Mark that the scene structure has been logged
        self._scene_structure_logged = True

    def _log_usd_subtree(self, prim_path: str) -> None:
        """Recursively log USD subtree starting from the given prim."""
        prim = self._scene.stage.GetPrimAtPath(prim_path)
        iterator = iter(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        for prim in iterator:
            # Skip guides
            if prim.GetAttribute("purpose").Get() == UsdGeom.Tokens.guide:
                # Skip descendants
                iterator.PruneChildren()
                continue
            # We're assuming that transforms below the rigid-body level are static
            log_usd_transform(prim, self._prev_usd_transforms)
            log_visuals(prim)

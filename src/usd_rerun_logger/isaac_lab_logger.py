import rerun as rr
import numpy as np


from .util import assert_usd_core_dependency, assert_isaac_lab_dependency

assert_usd_core_dependency()
assert_isaac_lab_dependency()



from .visual import log_visuals
from .transfom import log_usd_transform

from isaaclab.scene import InteractiveScene

from pxr import Usd, UsdGeom


class IsaacLabRerunLogger:
    def __init__(
        self,
        scene: "InteractiveScene",
        logged_envs: int | list[int] = 0,
    ):
        self._scene = scene
        self._last_transforms: dict[str, np.ndarray] = {}
        self._scene_structure_logged = False
        self._logged_envs = (
            [logged_envs] if isinstance(logged_envs, int) else logged_envs
        )

    def log_scene(self):
        if self._scene is None or self._scene.stage is None:
            return
        
        for obj in self._scene.articulations.values():
            poses = obj.data.body_pose_w.cpu().numpy()  # shape: (num_bodies, 3)
            for env_id in range(self._scene.num_envs):
                # Skip logging for unlisted environments
                if env_id not in self._logged_envs:
                    continue

                root_path = obj.cfg.prim_path.replace(".*", str(env_id))

                # Log the meshes once
                if not self._scene_structure_logged:
                    # Traverse the child prims to find mesh prims
                    prim = self._scene.stage.GetPrimAtPath(root_path)
                    for prim in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                        # We're assuming that transforms below the rigid-body level are static
                        log_usd_transform(prim)
                        log_visuals(prim)

                for body_index, body_name in enumerate(obj.body_names):
                    body_path = f"{root_path}/{body_name}"
                    pose = poses[env_id][body_index]

                    # Skip logging if the transform hasn't changed
                    if body_path in self._last_transforms and np.array_equal(
                        self._last_transforms[body_path], pose
                    ):
                        continue

                    self._last_transforms[body_path] = pose

                    rr.log(
                        body_path,
                        rr.Transform3D(
                            translation=pose[:3],
                            quaternion=(pose[4], pose[5], pose[6], pose[3]),
                        ),
                    )
        
        # Mark that the scene structure has been logged
        self._scene_structure_logged = True

from __future__ import annotations

import fnmatch
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectData, RigidObject
    from isaaclab.scene import InteractiveScene


import numpy as np
import rerun as rr
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import carb


class UsdRerunLogger:
    """Visualize USD stages in Rerun."""

    def __init__(self, use_physx_transforms: bool = False):
        """
        Initialize the UsdRerunLogger.
        Parameters
        ----------
        use_physx_transforms: bool
            Read the transform of prims with RigidBodyAPI from PhysX. This is useful when
            the USD stage is not updated during simulation (e.g., during training with Isaac Lab)
        """
        self._stage = None
        self._logger_id = None
        self._path_filter = None
        self._logged_meshes = set()  # Track which meshes we've already logged
        self._last_usd_transforms: dict[
            str, Gf.Matrix4d
        ] = {}  # Track last logged transforms for change detection
        self.use_physx_transforms = use_physx_transforms
        self._last_physx_transforms: dict[
            str, tuple[carb.Float3, carb.Float4]
        ] = {}  # Track last logged PhysX transforms for change detection
        self._isaac_lab_scene: "InteractiveScene" | None = None
        self._last_isaac_lab_transforms: dict[str, np.ndarray] = {}

    def initialize(
        self,
        stage: Usd.Stage,
        logger_id: str = "isaac_rerun_logger",
        spawn=True,
        save_path: Path | None = None,
        path_filter: str | list[str] | None = None,
        isaac_lab_scene: "InteractiveScene" | None = None,
    ):
        """
        Initialize the Rerun logger with a USD stage.

        Parameters
        ----------
        stage:
            The USD stage to log.
        logger_id:
            The ID of the Rerun logger.
        spawn:
            Whether to spawn the Rerun viewer.
        save_path:
            Path to save the Rerun recording to.
        path_filter:
            Glob pattern(s) to filter USD paths to log.
            Can be a single string or a list of strings.
            Example: "/World/Robot/*" or ["/World/Robot/*", "/World/Terrain"]
        """
        self.stop()
        self._stage = stage
        self._isaac_lab_scene = isaac_lab_scene

        if isinstance(path_filter, str):
            self._path_filter = [path_filter]
        else:
            self._path_filter = path_filter

        # Add random postfix to logger ID to avoid conflicts
        self._logger_id = f"{logger_id}_{np.random.randint(10000)}"
        rr.init(self._logger_id, spawn=spawn)
        if save_path is not None:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(save_path)

    def stop(self):
        """Stop the Rerun logger."""
        if self._logger_id is not None:
            rr.disconnect()
            self._logger_id = None
            self._stage = None
            self._path_filter = None
            self._logged_meshes.clear()
            self._last_usd_transforms.clear()
            self._last_physx_transforms.clear()
            self._last_isaac_lab_transforms.clear()

    def set_time(
        self,
        sequence: int | None = None,
        duration: int | float | timedelta | np.timedelta64 | None = None,
        timestamp: int | float | datetime | np.datetime64 | None = None,
    ) -> None:
        """
        Set the current time of the Rerun logger.

        Used for all subsequent logging on the same thread, until the next call to
        [`rerun.set_time`][], [`rerun.reset_time`][] or [`rerun.disable_timeline`][].

        For example: `set_time("frame_nr", sequence=frame_nr)`.

        There is no requirement of monotonicity. You can move the time backwards if you like.

        You are expected to set exactly ONE of the arguments `sequence`, `duration`, or `timestamp`.
        You may NOT change the type of a timeline, so if you use `duration` for a specific timeline,
        you must only use `duration` for that timeline going forward.

        Parameters
        ----------
        sequence:
            Used for sequential indices, like `frame_nr`.
            Must be an integer.
        duration:
            Used for relative times, like `time_since_start`.
            Must either be in seconds, a [`datetime.timedelta`][], or [`numpy.timedelta64`][].
            For nanosecond precision, use `numpy.timedelta64(nanoseconds, 'ns')`.
        timestamp:
            Used for absolute time indices, like `capture_time`.
            Must either be in seconds since Unix epoch, a [`datetime.datetime`][], or [`numpy.datetime64`][].
            For nanosecond precision, use `numpy.datetime64(nanoseconds, 'ns')`.

        """
        rr.set_time(
            "clock",
            sequence=sequence,
            duration=duration,
            timestamp=timestamp,
        )
    

    def log_stage(self):
        """
        Log the entire USD stage to Rerun.
        """
        if self._stage is None:
            print("Warning: USD stage is not initialized.")
            return

        # Traverse all prims in the stage
        current_paths = set()
        # Using Usd.TraverseInstanceProxies to traverse into instanceable prims (references)
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

        self._log_isaac_lab_asset_poses()

        for prim in self._stage.Traverse(predicate):
            # Skip guides
            if prim.GetAttribute("purpose").Get() == UsdGeom.Tokens.guide:
                continue

            entity_path = str(prim.GetPath())

            if self._path_filter:
                if not any(fnmatch.fnmatch(entity_path, p) for p in self._path_filter):
                    continue

            current_paths.add(entity_path)

            # Log transforms for all Xformable prims
            if entity_path not in self._last_isaac_lab_transforms:
                self._log_transform(prim, entity_path)

            # Log mesh geometry (only once per unique mesh)
            if prim.IsA(UsdGeom.Mesh):
                if entity_path not in self._logged_meshes:
                    self._log_mesh(prim, entity_path)
                    self._logged_meshes.add(entity_path)

            if prim.IsA(UsdGeom.Cube):
                if entity_path not in self._logged_meshes:
                    self._log_cube(prim, entity_path)
                    self._logged_meshes.add(entity_path)

        # Clear the logged paths that are no longer present in the stage
        for path in list(self._last_usd_transforms.keys()):
            if path not in current_paths:
                rr.log(path, rr.Clear.flat())
                del self._last_usd_transforms[path]


    def _log_transform(self, prim: Usd.Prim, entity_path: str):
        """Log transform using USD or PhysX API."""
        # Rigid Body transforms are read from PhysX if enabled
        if self.use_physx_transforms and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._log_physx_pose(prim, entity_path)
        else:
            self._log_usd_transform(prim, entity_path)

    def _log_isaac_lab_asset_poses(self):
        """
        Log all asset poses from the Isaac Lab scene to Rerun.
        """
        if self._isaac_lab_scene is None:
            return

        for name, obj in self._isaac_lab_scene.articulations.items():
            poses = obj.data.body_pose_w.cpu().numpy()  # shape: (num_bodies, 3)
            for env_id in range(self._isaac_lab_scene.num_envs):
                root_path = obj.cfg.prim_path.replace(".*", str(env_id))
                for body_index, body_name in enumerate(obj.body_names):
                    body_path = f"{root_path}/{body_name}"
                    pose = poses[env_id][body_index]

                    # Skip logging if the transform hasn't changed
                    if body_path in self._last_isaac_lab_transforms and np.array_equal(
                        self._last_isaac_lab_transforms[body_path], pose
                    ):
                        continue

                    self._last_isaac_lab_transforms[body_path] = pose

                    rr.log(body_path, rr.Transform3D(
                        translation=pose[:3],
                        quaternion=(pose[4], pose[5], pose[6], pose[3])
                    ))

    def _log_usd_transform(self, prim: Usd.Prim, entity_path: str):
        """Log the transform of an Xformable prim."""
        if not prim.IsA(UsdGeom.Xformable):
            return

        # Get the local transformation
        xformable = UsdGeom.Xformable(prim)
        transform_matrix: Gf.Matrix4d = xformable.GetLocalTransformation()

        if (
            entity_path in self._last_usd_transforms
            and self._last_usd_transforms[entity_path] == transform_matrix
        ):
            return

        self._last_usd_transforms[entity_path] = transform_matrix

        transform = Gf.Transform(transform_matrix)

        quaternion = transform.GetRotation().GetQuat()

        # Log the transform to Rerun
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=transform.GetTranslation(),
                quaternion=(*quaternion.GetImaginary(), quaternion.GetReal()),
                scale=transform.GetScale(),
            ),
        )

    def _log_physx_pose(self, prim: Usd.Prim, entity_path: str):
        """Log the PhysX transform of a prim."""
        from omni.physx import get_physx_interface

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return

        # Get the global transformation from PhysX
        transform = get_physx_interface().get_rigidbody_transformation(entity_path)
        if not transform["ret_val"]:
            return

        pos: carb.Float3 = transform["position"]
        rot: carb.Float4 = transform["rotation"]

        # Skip logging if the transform hasn't changed
        if entity_path in self._last_physx_transforms and self._last_physx_transforms[
            entity_path
        ] == (pos, rot):
            return
        self._last_physx_transforms[entity_path] = (pos, rot)

        # TODO: PhysX returns global transforms, and we currently don't handle if parent transforms exist.
        rr.log(entity_path, rr.Transform3D(translation=pos, quaternion=rot))

    

    

    def clear_logged_meshes(self):
        """Clear the cache of logged meshes, allowing them to be logged again."""
        self._logged_meshes.clear()


if __name__ == "__main__":
    test_usds = [
        # "/home/azazdeaz/repos/art/go2-example/assets/rail_blocks/rail_blocks.usd",
        # "/home/azazdeaz/repos/art/go2-example/assets/excavator_scan/excavator.usd",
        # "/home/azazdeaz/repos/art/go2-example/assets/stone_stairs/stone_stairs_f.usd",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/dex_cube_instanceable.usd",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_dex_cube_instanceable/dex_cube_instanceable.usda",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/simpleShading.usda",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_block_letter/block_letter_flat.usda",
        "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_go2-piamid/go2-piamid.usda",
    ]
    for usd_path in test_usds:
        print(f"\n\n\n>> Logging USD stage: {usd_path}")
        stage = Usd.Stage.Open(usd_path)
        logger = UsdRerunLogger(stage)
        logger.log_stage(frame_idx=0)

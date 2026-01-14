"""Rerun.io logger for Isaac Lab scenes."""

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rerun as rr

from .util import assert_isaac_lab_dependency, assert_usd_core_dependency
assert_usd_core_dependency()

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from .transfom import log_usd_transform  # noqa: E402
from .util import get_recording_stream  # noqa: E402
from .visual import log_visuals  # noqa: E402

# Note: In Isaac Lab, we can't read poses directly from the USD: https://github.com/isaac-sim/IsaacLab/issues/3472#issuecomment-3299713710

__all__ = [
    "IsaacLabRerunLogger",
]


class IsaacLabRerunLogger:
    """Logs Isaac Lab scenes to Rerun.io for visualization.

    This logger is specifically designed for Isaac Lab's :class:`InteractiveScene`,
    which contains articulations and rigid objects in a multi-environment setup.
    It reads body poses directly from Isaac Lab's physics simulation data rather
    than from USD transforms, ensuring accurate visualization of the simulated scene.

    The logger supports multi-environment Isaac Lab scenes and allows selecting
    which environment indices to log via the ``logged_envs`` parameter.

    The recording stream can either be provided directly via ``recording_stream``, or
    a new recording stream can be created that saves to ``save_path``. If neither is
    provided, it will try to find it using ``rr.get_data_recording()``. ``save_path``
    takes precedence over ``recording_stream``.

    Parameters
    ----------
    scene:
        The Isaac Lab :class:`InteractiveScene` to log. This scene contains the
        articulations and rigid objects that will be visualized.
    logged_envs:
        Indices of environments to log (for multi-env Isaac Lab scenes). If an ``int``,
        it is treated as a single environment index. If a list, those indices are logged.
        Defaults to ``0`` (the first environment).
    recording_stream:
        The Rerun recording stream to use. Ignored if ``save_path`` is provided.
    save_path:
        Path where the Rerun recording will be saved as an ``.rrd`` file. If provided,
        a new recording stream is created that saves to this path.
    application_id:
        Application ID for the Rerun recording. Used when creating a new recording
        stream (either via ``save_path`` or when falling back to a new stream).

    Attributes
    ----------
    scene : InteractiveScene
        The Isaac Lab scene being logged.
    recording_stream : rr.RecordingStream
        The Rerun recording stream used for logging.

    Notes
    -----
    - Body poses are read from Isaac Lab's simulation data (``body_pose_w``), not from
      USD transforms. This is because Isaac Lab updates physics internally without
      reflecting changes in the USD stage transforms.
    - Visual geometry (meshes) is logged only once on first encounter and cached.
    - Transforms are only re-logged when they change between calls to :meth:`log_scene`.
    - Scale information is preserved from the original USD transforms.
    - Prims with ``purpose`` set to ``guide`` are skipped along with their children.

    Examples
    --------
    Log an Isaac Lab scene directly:

    .. code-block:: python

       import rerun as rr
       from usd_rerun_logger import IsaacLabRerunLogger

       rr.init("isaac_lab_scene", spawn=True)
       logger = IsaacLabRerunLogger(scene)
       logger.log_scene()

    Log multiple environments:

    .. code-block:: python

       # Log environments 0, 1, and 2
       logger = IsaacLabRerunLogger(scene, logged_envs=[0, 1, 2])
       logger.log_scene()

    Save the recording to a file:

    .. code-block:: python

       logger = IsaacLabRerunLogger(scene, save_path="simulation.rrd")
       logger.log_scene()

    Continuous logging during simulation:

    .. code-block:: python

       rr.init("simulation", spawn=True)
       logger = IsaacLabRerunLogger(scene)

       for step in range(1000):
           # Step the simulation
           scene.step()
           # Log current state - only changed transforms are re-logged
           rr.set_time_sequence("step", step)
           logger.log_scene()

    See Also
    --------
    LogRerun : Gymnasium wrapper that uses this logger for episode recording.
    UsdRerunLogger : Logger for generic USD stages (non-Isaac Lab).
    """

    def __init__(
        self,
        scene: "InteractiveScene",
        logged_envs: int | list[int] = 0,
        recording_stream: rr.RecordingStream | None = None,
        save_path: Path | str | None = None,
        application_id: str | None = None,
    ):
        """Create the Isaac Lab Rerun logger."""
        # Ensure Isaac Lab dependencies are available
        assert_isaac_lab_dependency()

        self._scene = scene
        self._recording_stream = get_recording_stream(
            recording_stream=recording_stream,
            save_path=save_path,
            application_id=application_id,
        )
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

    @property
    def scene(self) -> "InteractiveScene":
        """The Isaac Lab scene being logged."""
        return self._scene

    @property
    def recording_stream(self) -> rr.RecordingStream:
        """The Rerun recording stream used for logging."""
        return self._recording_stream

    def log_scene(self):
        """Log the current state of the Isaac Lab scene to Rerun.

        Iterates over all articulations and rigid objects in the scene, reading
        their body poses from the simulation and logging them to Rerun. Visual
        geometry is logged only on the first call; subsequent calls only update
        transforms that have changed.

        This method is safe to call even if the scene or stage is ``None``; in
        that case, it returns immediately without logging anything.
        """
        if self.scene is None or self.scene.stage is None:
            return

        assets = itertools.chain(
            self.scene.articulations.items(),
            self.scene.rigid_objects.items(),
        )
        for obj_name, obj in assets:
            poses = obj.data.body_pose_w.cpu().numpy()  # shape: (num_bodies, 3)

            for env_id in range(self.scene.num_envs):
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
                    self.recording_stream.log(
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
        prim = self.scene.stage.GetPrimAtPath(prim_path)
        iterator = iter(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        for prim in iterator:
            # Skip guides
            if prim.GetAttribute("purpose").Get() == UsdGeom.Tokens.guide:
                # Skip descendants
                iterator.PruneChildren()
                continue
            # We're assuming that transforms below the rigid-body level are static
            log_usd_transform(self.recording_stream, prim, self._prev_usd_transforms)
            log_visuals(self.recording_stream, prim)

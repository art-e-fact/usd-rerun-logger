"""Rerun.io logger for any USD stage."""

import fnmatch
from pathlib import Path

from .util import assert_usd_core_dependency

assert_usd_core_dependency()

import rerun as rr  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from .transfom import log_usd_transform  # noqa: E402
from .util import get_recording_stream  # noqa: E402
from .visual import log_visuals  # noqa: E402

__all__ = [
    "UsdRerunLogger",
]


class UsdRerunLogger:
    """Logs USD (Universal Scene Description) stages to Rerun.io for visualization.

    This logger traverses a USD stage and logs all transforms and visual geometry
    (meshes, cubes, spheres, etc.) to a Rerun recording stream. It supports incremental
    logging, only re-logging transforms that have changed between calls to :meth:`log_stage`.

    The recording stream can either be provided directly via ``recording_stream``, or
    a new recording stream can be created that saves to ``save_path``. If neither is
    provided, it will try to find it using ``rr.get_data_recording()``. ``save_path``
    takes precedence over ``recording_stream``.

    Parameters
    ----------
    stage:
        The USD stage to log. This is the root of the scene hierarchy that will
        be traversed and logged.
    path_filter:
        Glob pattern(s) to filter which prims are logged. Can be a single pattern
        string or a list of patterns. Patterns starting with ``"!"`` are treated
        as exclusion patterns. For example, ``"/World/*"`` includes only prims
        under ``/World``, while ``"!/World/Hidden/*"`` excludes prims under
        ``/World/Hidden``. If ``None``, all prims are logged.
    recording_stream:
        The Rerun recording stream to use. Ignored if ``save_path`` is provided.
    save_path:
        Path where the Rerun recording will be saved as an ``.rrd`` file. If provided,
        a new recording stream is created that saves to this path.
    application_id:
        Application ID for the Rerun recording. Used when creating a new recording
        stream (either via ``save_path`` or when falling back to a new stream).
        Defaults to ``"usd_logger"`` if not provided.

    Attributes
    ----------
    stage : Usd.Stage
        The USD stage being logged.
    recording_stream : rr.RecordingStream
        The Rerun recording stream used for logging.

    Notes
    -----
    - Meshes are logged only once (on first encounter) and tracked internally to
      avoid redundant logging.
    - Transforms are logged every time :meth:`log_stage` is called, but only if
      they have changed since the last call.
    - Prims with ``purpose`` set to ``guide`` are skipped, along with their children.
    - Instance proxies (referenced/instanced prims) are traversed and logged.
    - When a prim is removed from the stage, it is automatically cleared from
      the Rerun recording.

    Examples
    --------
    Log a USD stage to the Rerun viewer:

    .. code-block:: python

       import rerun as rr
       from pxr import Usd
       from usd_rerun_logger import UsdRerunLogger

       rr.init("my_usd_viewer", spawn=True)
       stage = Usd.Stage.Open("my_scene.usda")
       logger = UsdRerunLogger(stage)
       logger.log_stage()

    Save the recording to a file:

    .. code-block:: python

       stage = Usd.Stage.Open("my_scene.usda")
       logger = UsdRerunLogger(stage, save_path="recording.rrd")
       logger.log_stage()

    Filter which prims are logged:

    .. code-block:: python

       # Log only prims under /World/Robots, but exclude /World/Robots/Debug
       logger = UsdRerunLogger(
           stage,
           path_filter=["/World/Robots/*", "!/World/Robots/Debug/*"]
       )
       logger.log_stage()

    Animate a scene by logging at each time step:

    .. code-block:: python

       rr.init("animated_scene", spawn=True)
       stage = Usd.Stage.Open("animated.usda")
       logger = UsdRerunLogger(stage)

       for frame in range(100):
           # Update USD stage time code
           stage.SetTimeCode(frame)
           # Log current state - only changed transforms are re-logged
           rr.set_time_sequence("frame", frame)
           logger.log_stage()
    """

    def __init__(
        self,
        stage: Usd.Stage,
        path_filter: str | list[str] | None = None,
        recording_stream: rr.RecordingStream | None = None,
        save_path: Path | str | None = None,
        application_id: str | None = None,
    ):
        """Create the USD Rerun logger."""
        self._stage = stage
        self._recording_stream = get_recording_stream(
            recording_stream=recording_stream,
            save_path=save_path,
            application_id=application_id,
        )
        self._logged_meshes = set()  # Track which meshes we've already logged
        self._last_usd_transforms: dict[
            str, Gf.Matrix4d
        ] = {}  # Track last logged transforms for change detection
        filters = (
            [path_filter] if isinstance(path_filter, str) else list(path_filter or [])
        )
        self._path_filter = filters or None
        include_filters: list[str] = []
        exclude_filters: list[str] = []
        for pattern in filters:
            if pattern.startswith("!") and len(pattern) > 1:
                exclude_filters.append(pattern[1:])
            else:
                include_filters.append(pattern)
        self._include_filter = include_filters or None
        self._exclude_filter = exclude_filters or None
        self._prev_transforms: dict[str, Gf.Matrix4d] = {}

    @property
    def stage(self) -> Usd.Stage:
        """The USD stage being logged."""
        return self._stage

    @property
    def recording_stream(self) -> rr.RecordingStream:
        """The Rerun recording stream used for logging."""
        return self._recording_stream

    def log_stage(self):
        """Log the current state of the USD stage to Rerun.

        Traverses all prims in the stage and logs their transforms and visual
        geometry. Transforms are only re-logged if they have changed since the
        last call. Meshes and other visual geometry are logged only once on
        first encounter.

        Prims with ``purpose`` set to ``guide`` are skipped along with their
        children. Path filters (if configured) are applied to include/exclude
        specific prim paths.

        When prims are removed from the stage between calls, they are
        automatically cleared from the Rerun recording.
        """
        # Traverse all prims in the stage
        current_paths = set()
        # Using Usd.TraverseInstanceProxies to traverse into instanceable prims (references)
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

        iterator = iter(self._stage.Traverse(predicate))
        for prim in iterator:
            # Skip guides
            if prim.GetAttribute("purpose").Get() == UsdGeom.Tokens.guide:
                iterator.PruneChildren()
                continue

            entity_path = str(prim.GetPath())

            # Apply path filters
            if self._include_filter and not any(
                fnmatch.fnmatch(entity_path, pattern)
                for pattern in self._include_filter
            ):
                continue
            if self._exclude_filter and any(
                fnmatch.fnmatch(entity_path, pattern)
                for pattern in self._exclude_filter
            ):
                continue

            current_paths.add(entity_path)

            # Log transforms for all Xformable prims
            log_usd_transform(self._recording_stream, prim, self._last_usd_transforms)

            if entity_path not in self._logged_meshes:
                # Log visuals for Mesh prims
                log_visuals(self._recording_stream, prim)
                self._logged_meshes.add(entity_path)

        # Clear the logged paths that are no longer present in the stage
        for path in list(self._last_usd_transforms.keys()):
            if path not in current_paths:
                self._recording_stream.log(path, rr.Clear.flat())
                del self._last_usd_transforms[path]

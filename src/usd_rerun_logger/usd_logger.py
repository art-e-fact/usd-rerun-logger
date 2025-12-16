import fnmatch

from .util import assert_usd_core_dependency

assert_usd_core_dependency()

import rerun as rr
from pxr import Gf, Usd, UsdGeom

from .transfom import log_usd_transform
from .visual import log_visuals


class UsdRerunLogger:
    def __init__(
        self,
        stage: Usd.Stage,
        path_filter: str | list[str] | None = None,
    ):
        """
        Docstring for __init__

        :param self: Description
        :param stage: Description
        :type stage: Usd.Stage
        :param path_filter: Glob pattern(s) to include; prefix with "!" to exclude.
        :type path_filter: str | list[str] | None
        """
        self._stage = stage
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

    def log_stage(self):
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
            log_usd_transform(prim, self._last_usd_transforms)

            if entity_path not in self._logged_meshes:
                # Log visuals for Mesh prims
                log_visuals(prim)
                self._logged_meshes.add(entity_path)

        # Clear the logged paths that are no longer present in the stage
        for path in list(self._last_usd_transforms.keys()):
            if path not in current_paths:
                rr.log(path, rr.Clear.flat())
                del self._last_usd_transforms[path]

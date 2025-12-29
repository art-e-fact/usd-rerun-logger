import rerun as rr
from pxr import Usd, UsdGeom

from .mesh import log_mesh
from .prim import log_cube


def log_visuals(recording_stream: rr.RecordingStream, prim: Usd.Prim):
    """Log visual geometry to Rerun."""
    if prim.IsA(UsdGeom.Mesh):
        log_mesh(recording_stream, prim)

    if prim.IsA(UsdGeom.Cube):
        log_cube(recording_stream, prim)

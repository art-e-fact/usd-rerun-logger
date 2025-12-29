from pxr import Usd, UsdGeom
from .mesh import log_mesh
from .prim import log_cube


def log_visuals(prim: Usd.Prim):
    """Log visual geometry to Rerun."""
    if prim.IsA(UsdGeom.Mesh):
        log_mesh(prim)

    if prim.IsA(UsdGeom.Cube):
        log_cube(prim)

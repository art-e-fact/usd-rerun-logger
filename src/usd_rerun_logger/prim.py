from pxr import Usd, UsdGeom
import rerun as rr


def log_cube(prim: Usd.Prim):
    """Log a cube prim as a Rerun box."""
    cube = UsdGeom.Cube(prim)
    entity_path = str(prim.GetPath())
    size_attr = cube.GetSizeAttr()
    size = size_attr.Get() if size_attr else 2.0
    half_size = size / 2.0

    color = None
    display_color_attr = cube.GetDisplayColorAttr()
    if display_color_attr and display_color_attr.HasValue():
        colors = display_color_attr.Get()
        if colors and len(colors) > 0:
            c = colors[0]
            color = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

    rr.log(
        entity_path,
        rr.Boxes3D(
            half_sizes=[half_size, half_size, half_size],
            colors=color,
            fill_mode="solid",
        ),
        static=True,
    )

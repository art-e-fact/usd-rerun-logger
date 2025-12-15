from pxr import Gf, Usd, UsdGeom
import rerun as rr


def log_usd_transform(prim: Usd.Prim):
    """Log the transform of an Xformable prim."""
    if not prim.IsA(UsdGeom.Xformable):
        return

    # Get the local transformation
    xformable = UsdGeom.Xformable(prim)
    transform_matrix: Gf.Matrix4d = xformable.GetLocalTransformation()

    transform = Gf.Transform(transform_matrix)

    quaternion = transform.GetRotation().GetQuat()

    # Log the transform to Rerun
    rr.log(
        str(prim.GetPath()),
        rr.Transform3D(
            translation=transform.GetTranslation(),
            quaternion=(*quaternion.GetImaginary(), quaternion.GetReal()),
            scale=transform.GetScale(),
        ),
    )
    print(f"    Logged transform for prim {prim.GetPath()}.")

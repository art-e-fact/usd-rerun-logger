from pxr import Gf, Usd, UsdGeom
import rerun as rr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import carb  # type: ignore


def log_usd_transform(
    prim: Usd.Prim, prev_transforms: dict[str, Gf.Matrix4d] | None = None
) -> None:
    """Log the transform of an Xformable prim."""
    if not prim.IsA(UsdGeom.Xformable):
        return

    # Get the local transformation
    xformable = UsdGeom.Xformable(prim)
    transform_matrix: Gf.Matrix4d = xformable.GetLocalTransformation()

    entity_path = str(prim.GetPath())

    # If previous transforms are provided, only log if there's a change
    if prev_transforms is not None:
        if (
            entity_path in prev_transforms
            and transform_matrix == prev_transforms[entity_path]
        ):
            # No change in transform; skip logging
            return
        else:
            # Update the previous transform
            prev_transforms[entity_path] = transform_matrix

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


def log_physx_pose(
    prim: Usd.Prim,
    entity_path: str,
    prev_transforms: dict[str, tuple["carb.Float3", "carb.Float4"]] | None = None,
) -> None:
    """Log the PhysX transform of a prim."""
    from omni.physx import get_physx_interface
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return

    # Get the global transformation from PhysX
    transform = get_physx_interface().get_rigidbody_transformation(entity_path)
    if not transform["ret_val"]:
        return

    pos: "carb.Float3" = transform["position"]
    rot: "carb.Float4" = transform["rotation"]

    # Skip logging if the transform hasn't changed
    if entity_path in prev_transforms and prev_transforms[entity_path] == (pos, rot):
        return
    prev_transforms[entity_path] = (pos, rot)

    # TODO: PhysX returns global transforms, and we currently don't handle if parent transforms exist.
    rr.log(entity_path, rr.Transform3D(translation=pos, quaternion=rot))

import os

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdShade

from usd_rerun_logger.util import download_image, is_url


def _load_texture(stage, texture_path):
    """Load texture from path."""
    if not texture_path:
        return None
    try:
        # Resolve path relative to stage
        if not os.path.isabs(texture_path):
            stage_path = stage.GetRootLayer().realPath
            if stage_path:
                texture_path = os.path.join(os.path.dirname(stage_path), texture_path)

        if not os.path.exists(texture_path):
            print(f"Warning: Texture file does not exist: {texture_path}")
            return None

        with Image.open(texture_path) as img:
            img = img.convert("RGB")  # Ensure 3 channels
            # mirror the image vertically and horizontally
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_data = np.array(img)
            return img_data

    except Exception as e:
        print(f"Failed to load texture {texture_path}: {e}")
        return None


def _resolve_asset_path(asset_path) -> str | None:
    """Resolve an Sdf.AssetPath to an absolute file path.

    Handles both local paths and URLs. For URLs, downloads the image
    to a cache directory and returns the local path.
    """
    if not isinstance(asset_path, Sdf.AssetPath):
        return None

    # TODO: Use Ar.GetResolver().OpenAsset() when it will be available
    path = asset_path.resolvedPath or asset_path.path
    if not path:
        print(f"Failed to resolve asset path: {asset_path}")
        return None

    if is_url(path):
        return download_image(path)

    if os.path.exists(path):
        return path

    path = asset_path.path
    if path:
        # Check if it's a URL
        if is_url(path):
            print(f"_resolve_asset_path: Downloading texture from URL: {path}")
            return download_image(path)

        if os.path.exists(path):
            return path

    return None


def _get_image_texture_path(prim: Usd.Prim) -> str | Gf.Vec3f | None:
    """
    Get material color or texture path.
    Returns: texture_path or None
    """
    binding_api = UsdShade.MaterialBindingAPI(prim)
    material: UsdShade.Material = binding_api.ComputeBoundMaterial()[0]
    if not material:
        print(f"No material found for prim {prim.GetPath()}.")
        return None

    shader: UsdShade.Shader = material.ComputeSurfaceSource()[0]

    # If no shader found, try to find connected shader from the material's mdl:surface output
    if not shader:
        mdl_surface = material.GetOutput("mdl:surface")
        if mdl_surface and mdl_surface.HasConnectedSource():
            source, sourceName, sourceType = mdl_surface.GetConnectedSource()
            shader = UsdShade.Shader(source)
        else:
            return None

    diffuse_color = shader.GetInput("diffuseColor")
    if diffuse_color:
        diffuse_color_source = diffuse_color.GetConnectedSource()

        if diffuse_color_source:
            diffuse_color_source: UsdShade.ConnectableAPI = diffuse_color_source[0]

            diffuse_color_source_file = diffuse_color_source.GetInput("file")
            diffuse_color_source_file_path = diffuse_color_source_file.Get()

            if not diffuse_color_source_file_path or not isinstance(
                diffuse_color_source_file_path, Sdf.AssetPath
            ):
                print("Diffuse color source is not a valid texture file path.")
                return None

            return _resolve_asset_path(diffuse_color_source_file_path)
        else:
            val = diffuse_color.Get()
            if val:
                if isinstance(val, Sdf.AssetPath):
                    return _resolve_asset_path(val)
                elif isinstance(val, Gf.Vec3f):
                    return val

    diffuse_texture = shader.GetInput("diffuse_texture")
    if diffuse_texture:
        # Check for connected source
        diffuse_texture_source = diffuse_texture.GetConnectedSource()
        if diffuse_texture_source:
            source, input_name, _ = diffuse_texture_source
            diffuse_texture_source_file = source.GetInput(input_name).Get()
            resolved_path = _resolve_asset_path(diffuse_texture_source_file)
            if resolved_path:
                return resolved_path

        # Check for direct value
        resolved_path = _resolve_asset_path(diffuse_texture.Get())
        if resolved_path:
            return resolved_path

    # Fallback to diffuse_color_constant if texture is missing or invalid
    diffuse_color_constant = shader.GetInput("diffuse_color_constant")
    if diffuse_color_constant:
        val = diffuse_color_constant.Get()
        if val and isinstance(val, Gf.Vec3f):
            return val

    diffuse_color_texture = shader.GetInput("base_color_texture")
    if diffuse_color_texture and diffuse_color_texture.HasConnectedSource():
        diffuse_texture_source = diffuse_color_texture.GetConnectedSource()[0]
        diffuse_texture_source_file: Sdf.AssetPath = diffuse_texture_source.GetInput(
            "texture"
        ).Get()
        resolved_path = _resolve_asset_path(diffuse_texture_source_file)
        if resolved_path:
            return resolved_path
        else:
            print("Diffuse texture source is not a valid texture file path.")

    print(f"Failed to parse color for prim {prim.GetPath()}.")
    return None


def extract_color_map(prim: Usd.Prim) -> tuple[np.ndarray | None, Gf.Vec3f | None]:
    """Extract color map from the material bound to the given prim."""
    texture_path = _get_image_texture_path(prim)
    if isinstance(texture_path, str):
        texture_buffer = _load_texture(prim.GetStage(), texture_path)
        return texture_buffer, None
    elif isinstance(texture_path, Gf.Vec3f):
        return None, texture_path
    else:
        return None, None

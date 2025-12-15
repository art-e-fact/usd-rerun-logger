from pxr import Usd, UsdShade, Sdf
from PIL import Image
import os
import numpy as np


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


def _get_image_texture_path(prim: Usd.Prim):
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
    if not shader:
        print("No surface shader found.")
        mdl_surface = material.GetOutput("mdl:surface")
        if mdl_surface and mdl_surface.HasConnectedSource():
            source, sourceName, sourceType = mdl_surface.GetConnectedSource()
            shader = UsdShade.Shader(source)
        else:
            return None

    implementation_source = shader.GetImplementationSource()

    if (
        implementation_source == "id"
        and shader.GetIdAttr().Get() == "UsdPreviewSurface"
    ):
        diffuse_color = shader.GetInput("diffuseColor")

        diffuse_color_source: UsdShade.ConnectableAPI = (
            diffuse_color.GetConnectedSource()[0]
        )

        diffuse_color_source_file = diffuse_color_source.GetInput("file")
        diffuse_color_source_file_path = diffuse_color_source_file.Get()

        if not diffuse_color_source_file_path or not isinstance(
            diffuse_color_source_file_path, Sdf.AssetPath
        ):
            print("Diffuse color source is not a valid texture file path.")
            return None

        return None, diffuse_color_source_file_path.resolvedPath

    elif (
        implementation_source == UsdShade.Tokens.sourceAsset
        and shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
        == "OmniPBR"
    ):
        diffuse_texture = shader.GetInput("diffuse_texture")
        if not diffuse_texture:
            print("No diffuse_texture input found in OmniPBR shader.")
            print("Shader inputs:", [inp.GetBaseName() for inp in shader.GetInputs()])
            return None
        print(diffuse_texture.GetConnectedSource())
        diffuse_texture_source = diffuse_texture.GetConnectedSource()
        if not diffuse_texture_source:
            print("No connected source for diffuse_texture.")
            return None
        diffuse_texture_source, input_name, _ = diffuse_texture_source
        diffuse_texture_source_file = diffuse_texture_source.GetInput(input_name).Get()
        if not diffuse_texture_source_file or not isinstance(
            diffuse_texture_source_file, Sdf.AssetPath
        ):
            print("Diffuse texture source is not a valid texture file path.")
            return None
        return diffuse_texture_source_file.resolvedPath

    elif (
        implementation_source == UsdShade.Tokens.sourceAsset
        and shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
        == "gltf_material"
    ):
        diffuse_texture = shader.GetInput("base_color_texture")
        print(diffuse_texture.GetConnectedSource())
        diffuse_texture_source = diffuse_texture.GetConnectedSource()[0]
        diffuse_texture_source_file: Sdf.AssetPath = diffuse_texture_source.GetInput(
            "texture"
        ).Get()
        if not diffuse_texture_source_file or not isinstance(
            diffuse_texture_source_file, Sdf.AssetPath
        ):
            print("Diffuse texture source is not a valid texture file path.")
            return None
        return diffuse_texture_source_file.resolvedPath

    else:
        print(f"Unsupported shader type: {shader.GetIdAttr().Get()}")
        return None


def extract_color_map(prim: Usd.Prim) -> np.ndarray | None:
    """Extract color map from the material bound to the given prim."""
    texture_path = _get_image_texture_path(prim)
    if texture_path:
        texture_buffer = _load_texture(prim.GetStage(), texture_path)
        return texture_buffer
    return None

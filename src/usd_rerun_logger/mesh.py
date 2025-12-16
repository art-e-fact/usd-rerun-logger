import numpy as np
from pxr import Usd, UsdGeom, Sdf, Tf
import rerun as rr

from .shader import extract_color_map


def log_mesh(prim: Usd.Prim):
    """Log mesh geometry to Rerun."""
    mesh = UsdGeom.Mesh(prim)
    entity_path = str(prim.GetPath())

    # Get vertex positions
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        return
    vertices = np.array(points_attr.Get())

    # Get face vertex indices
    face_vertex_indices_attr = mesh.GetFaceVertexIndicesAttr()
    face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()

    if not face_vertex_indices_attr or not face_vertex_counts_attr:
        rr.log(entity_path, rr.Points3D(positions=vertices), static=True)
        return

    face_vertex_indices = np.array(face_vertex_indices_attr.Get())
    face_vertex_counts = np.array(face_vertex_counts_attr.Get())

    if face_vertex_indices is None or face_vertex_counts is None:
        rr.log(entity_path, rr.Points3D(positions=vertices), static=True)
        return

    # --- Handle UVs ---
    # Use UsdGeom.PrimvarsAPI to handle indexed vs non-indexed primvars correctly
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    st_primvar = primvars_api.GetPrimvar("st")

    texcoords = None
    st_interpolation = "constant"

    if st_primvar:
        st_interpolation = st_primvar.GetInterpolation()

        # Get the data, resolving indices if present
        st_data = st_primvar.Get()
        st_indices = st_primvar.GetIndices()

        if st_data is not None:
            st_data = np.array(st_data)
            if st_indices:
                st_indices = np.array(st_indices)
                texcoords = st_data[st_indices]
            else:
                texcoords = st_data

    # --- Handle Normals ---
    normals_attr = mesh.GetNormalsAttr()
    normals = None
    normals_interpolation = "constant"
    if normals_attr.HasValue():
        normals = np.array(normals_attr.Get())
        normals_interpolation = normals_attr.GetMetadata("interpolation")

    # --- Flattening Logic ---
    # If UVs or Normals are face-varying, we must flatten the mesh to a triangle soup
    should_flatten = (st_interpolation == "faceVarying") or (
        normals_interpolation == "faceVarying"
    )

    # Fallback: if texcoords length matches face_vertex_indices length, treat as face-varying
    # (This handles cases where metadata might be missing or ambiguous but data shape is clear)
    if (
        texcoords is not None
        and len(texcoords) == len(face_vertex_indices)
        and len(texcoords) != len(vertices)
    ):
        should_flatten = True

    triangles_list = None

    # Map for subsets: face_index -> list of triangle_indices
    face_to_triangle_indices = [[] for _ in range(len(face_vertex_counts))]
    current_triangle_index = 0

    if should_flatten:
        # Flatten positions: Create a new vertex for every face corner
        vertices = vertices[face_vertex_indices]

        # Flatten normals if they are vertex-interpolated
        if normals is not None:
            if normals_interpolation == "vertex":
                normals = normals[face_vertex_indices]
            # if faceVarying, normals should already match face_vertex_indices length

        # Flatten UVs if they are vertex-interpolated
        if texcoords is not None:
            if st_interpolation == "vertex":
                texcoords = texcoords[face_vertex_indices]
            # if faceVarying, texcoords should already match face_vertex_indices length

        # Generate trivial triangles (0,1,2), (3,4,5)...
        # But we must respect the polygon counts (3, 4, etc.)
        triangles = []
        idx = 0
        for face_idx, count in enumerate(face_vertex_counts):
            # The vertices for this face are at indices [idx, idx+1, ... idx+count-1] in our new arrays
            if count == 3:
                triangles.extend([idx, idx + 1, idx + 2])
                face_to_triangle_indices[face_idx].append(current_triangle_index)
                current_triangle_index += 1
            elif count == 4:
                triangles.extend([idx, idx + 1, idx + 2])
                face_to_triangle_indices[face_idx].append(current_triangle_index)
                current_triangle_index += 1

                triangles.extend([idx, idx + 2, idx + 3])
                face_to_triangle_indices[face_idx].append(current_triangle_index)
                current_triangle_index += 1
            else:
                # Fan triangulation
                for i in range(1, count - 1):
                    triangles.extend([idx, idx + i, idx + i + 1])
                    face_to_triangle_indices[face_idx].append(current_triangle_index)
                    current_triangle_index += 1
            idx += count

        triangles_list = np.array(triangles, dtype=np.uint32).reshape(-1, 3)

    else:
        # Standard indexed mesh path (shared vertices)
        triangles = []
        idx = 0
        for face_idx, count in enumerate(face_vertex_counts):
            if count == 3:
                triangles.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 1],
                        face_vertex_indices[idx + 2],
                    ]
                )
                face_to_triangle_indices[face_idx].append(current_triangle_index)
                current_triangle_index += 1
            elif count == 4:
                triangles.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 1],
                        face_vertex_indices[idx + 2],
                    ]
                )
                face_to_triangle_indices[face_idx].append(current_triangle_index)
                current_triangle_index += 1

                triangles.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 2],
                        face_vertex_indices[idx + 3],
                    ]
                )
                face_to_triangle_indices[face_idx].append(current_triangle_index)
                current_triangle_index += 1
            else:
                for i in range(1, count - 1):
                    triangles.extend(
                        [
                            face_vertex_indices[idx],
                            face_vertex_indices[idx + i],
                            face_vertex_indices[idx + i + 1],
                        ]
                    )
                    face_to_triangle_indices[face_idx].append(current_triangle_index)
                    current_triangle_index += 1
            idx += count

        triangles_list = np.array(triangles, dtype=np.uint32).reshape(-1, 3)

    # --- Material and Texture Handling ---
    texture_buffer = None

    subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
    if subsets:
        for subset in subsets:
            if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                print(
                    "Warning: Unsupported subset element type:",
                    subset.GetElementTypeAttr().Get(),
                )
                continue

            # Rearrange the mesh data to only include the subset
            included_faces = subset.GetIndicesAttr().Get()
            if not included_faces:
                continue

            # Collect all triangles for these faces
            subset_triangle_indices = []
            for face_idx in included_faces:
                if face_idx < len(face_to_triangle_indices):
                    subset_triangle_indices.extend(face_to_triangle_indices[face_idx])

            if not subset_triangle_indices:
                continue

            subset_triangles = triangles_list[subset_triangle_indices]

            # TODO: Remove unused vertices

            texture_buffer, color = extract_color_map(subset.GetPrim())

            rr.log(
                str(subset.GetPath()),
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=subset_triangles,
                    vertex_normals=normals,
                    vertex_texcoords=texcoords,
                    albedo_texture=texture_buffer,
                    albedo_factor=color,
                ),
                static=True,
            )

    else:
        texture_buffer, color = extract_color_map(prim)

        rr.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles_list,
                vertex_normals=normals,
                vertex_texcoords=texcoords,
                albedo_texture=texture_buffer,
                albedo_factor=color,
            ),
            static=True,
        )

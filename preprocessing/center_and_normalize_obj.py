import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import os
import argparse


def center_and_normalize(verts):
    """Center and scale vertices so that the mesh fits in a unit sphere."""
    # Compute bounding box
    min_coords = verts.min(dim=0)[0]
    max_coords = verts.max(dim=0)[0]
    center = (max_coords + min_coords) / 2
    
    # Center vertices
    verts_centered = verts - center
    
    # Scale to unit cube
    scale = (max_coords - min_coords).max()
    verts_normalized = verts_centered / scale
        
    return verts_normalized

def save_obj_with_textures(
    filename, 
    verts, 
    faces, 
    aux, 
    mtl_filename="material.mtl"
):
    """
    Save an OBJ with positions, UVs, and per-face materials. 
    Assumes all textures/materials come from the original MTL.
    """
    # Unpack needed data
    verts_idx = faces.verts_idx
    tex_idx   = faces.textures_idx
    materials = list(aux.material_colors.keys())
    mat_idx   = faces.materials_idx  # Face -> material index

    # Write .obj
    with open(filename, "w") as f:
        # Reference the same MTL
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n")

        # Write vertex positions
        for x, y, z in verts:
            f.write(f"v {x.item()} {y.item()} {z.item()}\n")

        # Write UVs
        if aux.verts_uvs is not None:
            for u, v in aux.verts_uvs:
                f.write(f"vt {u.item()} {v.item()}\n")

        # Group faces by material, then write them
        current_mat = None
        for i in range(len(verts_idx)):
            mat_name = materials[mat_idx[i]]
            if mat_name != current_mat:
                f.write(f"usemtl {mat_name}\n")
                current_mat = mat_name

            v1, v2, v3 = (verts_idx[i] + 1).tolist()
            t1, t2, t3 = (tex_idx[i] + 1).tolist()
            f.write(f"f {v1}/{t1} {v2}/{t2} {v3}/{t3}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center and normalize a mesh.")
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file (e.g., .obj).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the normalized mesh.')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 1. Load source mesh with textures
    verts, faces, aux = load_obj(args.mesh_path, load_textures=True)
    
    # 2. Center and normalize
    verts_normalized = center_and_normalize(verts)

    # 3. Save the new mesh, reusing the original MTL/textures
    save_obj_with_textures(
        f"{args.output_path}/{os.path.basename(args.mesh_path)}", 
        verts_normalized, 
        faces, 
        aux, 
        mtl_filename="material.mtl"  # or wherever your .mtl resides
    )

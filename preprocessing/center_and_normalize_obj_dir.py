import numpy as np
import torch
from pytorch3d.io import load_obj
import os
import argparse

def center_and_normalize_all(verts, global_center, global_scale):
    """Normalize vertices using the global center and scale"""
    verts_centered = verts - global_center
    verts_normalized = verts_centered * global_scale  
      
    return verts_normalized

def rotate_verts(verts):
    # rotation should be -90 degrees around the x-axis and -90 degrees around the y-axis
    theta_x = -np.pi / 2  # -90 degrees
    theta_y = -np.pi / 2  # -90 degrees
    
    # Rotation matrices
    rotation_x = torch.tensor([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=verts.device)

    # rotation_y = torch.tensor([
    #     [np.cos(theta_y), 0, np.sin(theta_y), 0],
    #     [0, 1, 0, 0],
    #     [-np.sin(theta_y), 0, np.cos(theta_y), 0],
    #     [0, 0, 0, 1]
    # ], dtype=torch.float32, device=verts.device)

    # transformation = rotation_y @ rotation_x

    verts_h = torch.cat([verts, torch.ones((verts.shape[0], 1), device=verts.device)], dim=1)
    verts_rotated = verts_h @ rotation_x.T
    return verts_rotated[:, :3]  # Return only the first 3 columns (x, y, z)

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
    verts_idx = faces.verts_idx
    tex_idx   = faces.textures_idx
    materials = list(aux.material_colors.keys())
    mat_idx   = faces.materials_idx  # Face -> material index

    with open(filename, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n")
        for x, y, z in verts:
            f.write(f"v {x.item()} {y.item()} {z.item()}\n")
        if aux.verts_uvs is not None:
            for u, v in aux.verts_uvs:
                f.write(f"vt {u.item()} {v.item()}\n")
        current_mat = None
        for i in range(len(verts_idx)):
            mat_name = materials[mat_idx[i]]
            if mat_name != current_mat:
                f.write(f"usemtl {mat_name}\n")
                current_mat = mat_name
            v1, v2, v3 = (verts_idx[i] + 1).tolist()
            t1, t2, t3 = (tex_idx[i] + 1).tolist()
            f.write(f"f {v1}/{t1} {v2}/{t2} {v3}/{t3}\n")
            
def save_obj(filename, verts, faces):
    """
    Save an OBJ file with only positions and faces (no textures or materials).
    
    Args:
        filename: Path to output OBJ file
        verts: Tensor of vertex positions (N, 3)
        faces: Faces object with verts_idx attribute, or tensor of face indices
    """
    # Handle different face formats
    if hasattr(faces, 'verts_idx'):
        verts_idx = faces.verts_idx
    else:
        verts_idx = faces
    
    with open(filename, "w") as f:
        # Write vertices
        for x, y, z in verts:
            f.write(f"v {x.item()} {y.item()} {z.item()}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in verts_idx:
            v1, v2, v3 = (face + 1).tolist()
            f.write(f"f {v1} {v2} {v3}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Center and normalize all meshes in a directory together."
    )
    parser.add_argument('--mesh_path', type=str, required=True, 
                        help='Directory containing mesh files (e.g., .obj).')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Directory to store normalized mesh files.')
    args = parser.parse_args()

    if not os.path.isdir(args.mesh_path):
        raise ValueError(f"Input directory {args.mesh_path} does not exist.")
    os.makedirs(args.output_path, exist_ok=True)

    mesh_filenames = [f for f in os.listdir(args.mesh_path) if f.endswith(".obj")]
    mesh_filenames = sorted(mesh_filenames, key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))  # Sort by numeric suffix if present
    
    # First pass: load all meshes to compute global bounding box
    all_verts = []
    meshes_data = []  # store the loaded meshes to avoid re-loading later
    for file in mesh_filenames:
        input_file_path = os.path.join(args.mesh_path, file)
        print(f"Loading {input_file_path} for global normalization...")
        verts, faces, aux = load_obj(input_file_path, load_textures=True)
        # verts = rotate_verts(verts)
        all_verts.append(verts)
        meshes_data.append((file, verts, faces, aux))
    
    # Stack all vertices to find global bounding box
    concatenated_verts = torch.cat(all_verts, dim=0)
    global_min = concatenated_verts.min(dim=0)[0]
    global_max = concatenated_verts.max(dim=0)[0]
    global_center = (global_max + global_min) / 2
    global_scale = (global_max - global_min).max()
    global_scale = 1 / global_scale if global_scale > 0 else 1.0  # Avoid division by zero
    print(f"Global center: {global_center}, Global scale: {global_scale}")

    # Second pass: process and save normalized meshes
    for index, (file, verts, faces, aux) in enumerate(meshes_data):
        input_file_path = os.path.join(args.mesh_path, file)
        print(f"Processing {input_file_path}...")

        verts_normalized = center_and_normalize_all(verts, global_center, global_scale)

        name, ext = os.path.splitext(file)
        output_filename = f"{index:04d}{ext}"
        output_file_path = os.path.join(args.output_path, output_filename)

        dst_mtl_filename = f"{index:04d}.mtl"
        dst_texture_filename = f"{index}.png"
        save_obj_with_textures(
            output_file_path, 
            verts_normalized, 
            faces, 
            aux=aux,
            mtl_filename=dst_mtl_filename
        )
        print(f"Saved normalized mesh to {output_file_path}")

        src_mtl_file_path = input_file_path.replace(ext, ".mtl")
        output_mtl_file_path = os.path.join(args.output_path, dst_mtl_filename)
        if os.path.exists(src_mtl_file_path):
            os.system(f"cp {src_mtl_file_path} {output_mtl_file_path}")
            print(f"Copied MTL file to {output_mtl_file_path}")
        else:
            print(f"MTL file {src_mtl_file_path} not found. Skipping copy.")
            
        src_texture_file_path = input_file_path.replace(ext, ".png")
        output_texture_file_path = os.path.join(args.output_path, dst_texture_filename)
        if os.path.exists(src_texture_file_path):
            os.system(f"cp {src_texture_file_path} {output_texture_file_path}")
            print(f"Copied texture file to {output_texture_file_path}")
        else:
            print(f"Texture file {src_texture_file_path} not found. Skipping copy.")
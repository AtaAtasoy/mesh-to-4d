import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
import argparse

def is_mesh_inside_unit_sphere(mesh: Meshes, radius: float = 0.5) -> bool:
    """
    Check if all vertices of the mesh are inside a unit sphere with the given radius.
    """
    vertices = mesh.verts_packed()
    
    distances = torch.norm(vertices, dim=1)
    
    # Check if all distances are less than or equal to the radius
    is_inside = torch.all(distances <= radius).item()
    if not is_inside:
        print(f"Mesh vertices exceed the radius of {radius}.")
        print(f"Max distance: {distances.max().item()}")
    return is_inside

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if a mesh is inside a unit sphere.")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = load_objs_as_meshes([args.mesh_path], device=device)
    
    inside = is_mesh_inside_unit_sphere(mesh, radius=0.5)
    print(f"Is the mesh inside the unit sphere with radius 0.5? {inside}")
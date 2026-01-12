import os.path as osp
import open3d as o3d
from glob import glob


def load_scene_meshes(vid: str, storage_dir: str) -> dict:
    """
    Args:
        vid (str): Video ID in the format 'pid-vid'.
        storage_dir (str): path to <storage_dir>/Digital-Twin/meshes/<pid>/
        
    Returns:
        dict from mesh name to Open3D triangle mesh.
    """
    pid = vid.split('-')[0]
    mesh_dir = osp.join(storage_dir, "Digital-Twin/meshes", pid)
    mesh_paths = glob(osp.join(mesh_dir, f'{pid}*.obj'))
    if len(mesh_paths) == 0:
        raise ValueError(f"No meshes found in {mesh_dir}")

    geometries = {}
    for mesh_path in mesh_paths:
        mesh_name = osp.basename(mesh_path).split('.')[0]
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.is_empty():
            mesh.compute_vertex_normals()
            geometries[mesh_name] = mesh
        else:
            print(f"Warning: Mesh {mesh_name} is empty or could not be read.")
    return geometries
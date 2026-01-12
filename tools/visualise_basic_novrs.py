""" Visualise camera poses and meshes without VRS data.
This uses the intermediate_data/

Semi-dense points will not be displayed as they are not saved.
SlamL also not supported.
"""
from pathlib import Path
import open3d as o3d
import numpy as np
from argparse import ArgumentParser

import os.path as osp

from novrs_lib import NoVRSReader
from hovering_lib.helper import get_frustum

from glob import glob


""" Visualize poses of rgb (pinholecw90) frames. """


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--vid', type=str, help='video id')
    parser.add_argument('--show-mesh-frame', default=False,
                        help="To display the world xyz axis")
    parser.add_argument('--specify-frame-num', type=int, default=600)
    parser.add_argument('--num-display-poses', type=int, default=1000,
        help='uniformly sample num-display-poses to avoid creating too many poses')
    parser.add_argument('--frustum-size', type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frustum_size = args.frustum_size

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # The meshframe has size 1.0 metre

    reader = NoVRSReader(
        vid=args.vid,
        frame_type='mp4',
        load_pts=False,
        load_frame_traj=True)

    # pcd = reader.load_filtered_points(
    #     threshold_dep=args.threshold_dep,
    #     threshold_invdep=args.threshold_invdep, as_open3d=True)
    rgb_poses = reader.load_pinholecw90_trajectory()

    # load cupboards
    pid = args.vid.split('-')[0]
    mesh_dir = osp.join(reader.storage_dir, "Digital-Twin/meshes", pid)
    file_list = glob(osp.join(mesh_dir, f'{pid}*.obj'))
    print(f"Found {len(file_list)} meshes in {mesh_dir}")
    file_list.sort()
    geometry_list = []
    for fn in file_list:
        print(fn)
        mesh = o3d.io.read_triangle_mesh(fn)
        mesh.compute_vertex_normals()
        geometry_list.append(mesh)

    # use for gaze/surface ray intersection
    scene = o3d.t.geometry.RaycastingScene()

    object_ids = []
    for mesh in geometry_list:
        vis.add_geometry(mesh, reset_bounding_box=True)
        object_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        object_ids.append(object_id)
    print(object_ids)

    cam_h, cam_w = 1408, 1408
    if args.specify_frame_num is None:
        T_slamL_to_rgb = np.linalg.inv(reader.T_rgb_to_slamL)
        frustums = [
            get_frustum(c2w, sz=frustum_size, camera_height=cam_h, camera_width=cam_w)
            for c2w in rgb_poses]
        for frustum in frustums:
            vis.add_geometry(frustum, reset_bounding_box=True)
    else:
        frame_num = args.specify_frame_num
        c2w = rgb_poses[frame_num]
        assert c2w is not None, f"Frame {frame_num} has pose None."
        frustum = get_frustum(c2w, sz=frustum_size, camera_height=cam_h, camera_width=cam_w)
        vis.add_geometry(frustum, reset_bounding_box=True)

    # vis.add_geometry(pcd, reset_bounding_box=True)
    if args.show_mesh_frame:
        print("Show Mesh Frame")
        vis.add_geometry(mesh_frame, reset_bounding_box=True)

    control = vis.get_view_control()
    control.set_front([1, 1, 1])
    control.set_lookat([0, 0, 0])
    control.set_up([0, 0, 1])
    control.set_zoom(1)

    vis.run()
    vis.destroy_window()
from typing import List, Tuple, Union
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from collections import namedtuple
from open3d.visualization import rendering

from hovering_lib.o3d_line_mesh import LineMesh


geomType = namedtuple('geom3D', ['points', 'lines'])


def get_material_str(color: str, 
                     point_size=None, 
                     shader="defaultUnlit",
                     alpha=1.0) -> rendering.MaterialRecord:
    """
    Args:
        shader: e.g.'defaultUnlit', 'defaultLit', 'depth', 'normal',
            'unlitLine'
            see Open3D: cpp/open3d/visualization/rendering/filament/FilamentScene.cpp#L1109
    """
    base_colors = {
        'white': [1, 1, 1],
        'red': [1, 0, 0],
        'blue': [0, 0, 1],
        'light_blue': [0.65, 0.74, 0.86],
        'green': [0, 1, 0],
        'yellow': [1, 1, 0],
        'gray': [0.8, 0.8, 0.8],
        'purple': [0.2, 0.2, 0.8],
        'cyan': [0, 1, 1],
    }
    material = rendering.MaterialRecord()
    material.shader = shader
    material.base_color = base_colors[color] + [alpha]
    if point_size is not None:
        material.point_size = point_size
    return material

def get_material_rgb(rgb: Tuple, 
                     point_size=None, 
                     shader="defaultUnlit",
                     alpha=1.0) -> rendering.MaterialRecord:
    """
    Args:
        rgb: Tuple, RGB color in range [0, 255]
        shader: e.g.'defaultUnlit', 'defaultLit', 'depth', 'normal',
            'unlitLine', 'defaultLitTransparency', 'defaultUnlitTransparency'
            see Open3D: cpp/open3d/visualization/rendering/filament/FilamentScene.cpp#L1109
    """
    material = rendering.MaterialRecord()
    material.shader = shader
    base_color = [c / 255 for c in rgb] + [alpha]
    material.base_color = base_color
    if point_size is not None:
        material.point_size = point_size
    return material


def get_cam_pos(c2w: np.ndarray) -> np.ndarray:
     """ Get camera position in world coordinate system
     """
     cen = np.float32([0, 0, 0, 1])
     pos = c2w @ cen
     return pos[:3]


""" Source: see COLMAP """
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def get_c2w(img_data: list) -> np.ndarray:
    """
    Args:
        img_data: list, [qvec, tvec] of w2c
    
    Returns:
        c2w: np.ndarray, 4x4 camera-to-world matrix
    """
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(img_data[:4])
    w2c[:3, -1] = img_data[4:7]
    c2w = np.linalg.inv(w2c)
    return c2w

def get_frustum(c2w: np.ndarray,
                sz=0.2, 
                camera_height=None,
                camera_width=None,
                frustum_color=[1, 0, 0],
                to_geomtype=False) -> Union[o3d.geometry.LineSet, geomType]:
    """
    Args:
        c2w: np.ndarray, 4x4 camera-to-world matrix
        sz: float, size (width) of the frustum
    Returns:
        frustum: o3d.geometry.TriangleMesh
    """
    cen = [0, 0, 0]
    wid = sz
    if camera_height is not None and camera_width is not None:
        hei = wid * camera_height / camera_width
    else:
        hei = wid
    tl = [wid, hei, sz]
    tr = [-wid, hei, sz]
    br = [-wid, -hei, sz]
    bl = [wid, -hei, sz]
    points = np.float32([cen, tl, tr, br, bl])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],]

    if to_geomtype:
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points = (c2w @ points_h.T).T[:, :3]
        frustum = geomType(points=points, lines=lines)
    else:
        frustum = o3d.geometry.LineSet()
        frustum.points = o3d.utility.Vector3dVector(points)
        frustum.lines = o3d.utility.Vector2iVector(lines)
        frustum.colors = o3d.utility.Vector3dVector([np.asarray([1, 0, 0])])
        frustum.paint_uniform_color(frustum_color)
        frustum = frustum.transform(c2w)

    return frustum


def get_trajectory(pos_history,
                   num_line=6,
                   line_radius=0.15
                   ) -> o3d.geometry.TriangleMesh:
    """ pos_history: absolute position history
    """
    pos_history = np.asarray(pos_history)[-num_line:]
    colors = [0, 0, 0.6]
    line_mesh = LineMesh(
        points=pos_history, 
        colors=colors, radius=line_radius)
    line_mesh.merge_cylinder_segments()
    path = line_mesh.cylinder_segments[0]
    return path


def get_pretty_trajectory(pos_history,
                          num_line=6,
                          line_radius=0.15,
                          darkness=1.0,
                          ) -> List[o3d.geometry.TriangleMesh]:
    """ pos_history: absolute position history
    """
    def generate_jet_colors(n, darkness=0.6):
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0, vmax=n-1)
        colors = cmap(norm(np.arange(n)))
        # Convert RGBA to RGB
        colors_rgb = []
        for color in colors:
            colors_rgb.append(color[:3] * darkness)

        return colors_rgb

    pos_history = np.asarray(pos_history)[-num_line:]
    colors = generate_jet_colors(len(pos_history), darkness)
    line_mesh = LineMesh(
        points=pos_history, 
        colors=colors, radius=line_radius)
    return line_mesh.cylinder_segments


""" Obtain Viewpoint from Open3D GUI """
def parse_o3d_gui_view_status(status: dict, render: rendering.OffscreenRenderer):
    """ Parse open3d GUI's view status and convert to OffscreenRenderer format.
    This will do the normalisation of front and compute eye vector (updated version of front)

    
    Args:
        status: Ctrl-C output from Open3D GUI
        render: OffscreenRenderer
    Output:
       params for render.setup_camera(fov, lookat, eye, up) 
    """
    cam_info = status['trajectory'][0]
    fov = cam_info['field_of_view']
    lookat = np.asarray(cam_info['lookat'])
    front = np.asarray(cam_info['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(cam_info['up'])
    zoom = cam_info['zoom']
    """ 
    See Open3D/cpp/open3d/visualization/visualizer/ViewControl.cpp#L243: 
        void ViewControl::SetProjectionParameters()
    """
    right = np.cross(up, front) / np.linalg.norm(np.cross(up, front))
    view_ratio = zoom * render.scene.bounding_box.get_max_extent()
    distance = view_ratio / np.tan(fov * 0.5 / 180.0 * np.pi)
    eye = lookat + front * distance
    return fov, lookat, eye, up


def set_offscreen_as_gui(render: rendering.OffscreenRenderer, status: dict):
    """ Set offscreen renderer as GUI's view status
    """
    fov, lookat, eye, up = parse_o3d_gui_view_status(status, render)
    render.setup_camera(fov, lookat, eye, up)
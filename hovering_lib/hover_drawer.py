import os.path as osp
from typing import Tuple, List, Dict
from functools import lru_cache
import numpy as np
import open3d as o3d
import json
from open3d.visualization import rendering

from novrs_lib.scene_mesh import load_scene_meshes
from novrs_lib.novrs_reader_basic import NoVRSReader
from hovering_lib.colors import code_to_rgb
from hovering_lib.cv2_primitive_drawer import CV2PrimitiveDrawer
from hovering_lib.helper import (
    get_material_rgb, 
    get_frustum,
    set_offscreen_as_gui,
)


class HoverDrawer:

    """ Drawer for hovering cameras.
    """

    background_color = [1, 1, 1, 1.0]

    def __init__(self, 
                 reader: NoVRSReader,
                 out_width: int = 1280,
                 hover_draw_config: Dict = None,
                 ):
        """
        Args:
            reader
            out_width: int,
                e.g. 640 (height = 360), 1280 (height = 720), 1920 (height = 1080)
        """
        self.reader = reader

        # out_size need to be divisible by 2 for FFmpeg compatibility
        out_height = int(out_width * 9 / 16)
        out_height = out_height if out_height % 2 == 0 else out_height + 1
        self.out_size = (out_width, out_height)
        self.render = rendering.OffscreenRenderer(*self.out_size)

        """ Set up camera poses"""
        self.c2ws = reader.load_pinholecw90_trajectory()
        self._load_geometries()
    
        if hover_draw_config is None:
            hover_draw_config = {
                'frustum_size': 0.1, # self.reader.frustum_size,
                'point_size': 0.1,
                # 'frustum_line_width': 1,
                # 'traj_line_radius': 0.02,
                'frustum_line_width_cv2': 1,
                'traj_line_radius_cv2': 1
            }
        self.set_render_params(**hover_draw_config)
        self.set_render_params_cv2(**hover_draw_config)

        self.base_scene_img = None
        self.cv2_render = None

    def _load_geometries(self):
        """ Load geometries and viewstatus for Open3D visualisation 

        Digital Twin and cupboard viewpoints are needed.
        """
        reader = self.reader
        geometries = load_scene_meshes(reader.vid, reader.storage_dir)
        viewstatus_path = osp.join(
           f'./cupboard_viewpoints/{reader.pid}.json')
        with open(viewstatus_path) as f:
            o3d_viewstatus = json.load(f)
        self.geometries = geometries
        self.viewstatus = o3d_viewstatus
    
    def set_render_params(self,
                          point_size: float = 0.1,
                          frustum_size: float = 0.5,
                        #   frustum_line_width: float = 1,
                        #   traj_line_radius: float = 0.02,
                          *args, **kwargs
                          ):
        """
        This function is auto called at initialization.
        """
        self.point_size = point_size
        # Render Layout params
        self.frustum_size = frustum_size

        self.traj_mt = self.get_material('white')
    
    def set_render_params_cv2(self,
                            frustum_line_width_cv2: float = 1,
                            traj_line_radius_cv2: float = 1,
                            *args, **kwargs
                            ):
        """
        This function is auto called at initialization.
        """
        self.frustum_line_width_cv2 = frustum_line_width_cv2
        self.traj_line_radius_cv2 = traj_line_radius_cv2
    
    @lru_cache(maxsize=32)
    def get_material(self, 
                     color: str, 
                     point_size=None, 
                     line_width=None,
                     shader=None,
                     alpha=1.0):
        """
        Get material for rendering
        """
        mt = get_material_rgb(code_to_rgb(color), alpha=alpha)
        if point_size is not None:
            mt.point_size = point_size
        if line_width is not None:
            mt.line_width = line_width
        if shader is not None:
            mt.shader = shader
        return mt
    
    def render_base_scene(self, 
                          clear_geometry: bool = True,
                          sun_light: bool = False):
        """
        This shall be run once at the beginning.

        For self.geometries,
        - Pointcloud will be coloured white
        - Meshes will be coloured gray/light-blue

        """
        if clear_geometry:
            self.render.scene.clear_geometry()

        for name, geometry in self.geometries.items():
            if isinstance(geometry, o3d.geometry.PointCloud):
                mt = self.get_material('white', point_size=self.point_size)
            elif isinstance(geometry, o3d.geometry.TriangleMesh):
                mt = self.get_material(
                    'gray', point_size=self.point_size,
                    shader='defaultLitTransparency', alpha=0.5)
            else:
                raise TypeError(
                    f"Unsupported geometry type: {type(geometry)}. "
                    "Only PointCloud and TriangleMesh are supported.")
            self.render.scene.add_geometry(name, geometry, mt)

        # viewcontrol is valid only after putting on the scene
        if 'auto_view' in self.viewstatus:
            vs = self.viewstatus  # customary created viewstatus
            self.render.setup_camera(vs['fov'], vs['lookat'], vs['eye'], vs['up'])
        else:
            set_offscreen_as_gui(self.render, self.viewstatus)  # viewstatus from GUI
        self.render.scene.set_background(self.background_color)

        if sun_light:
            self.render.scene.scene.set_sun_light(
                [0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
            self.render.scene.scene.enable_sun_light(True)
        else:
            self.render.scene.set_lighting(
                rendering.Open3DScene.NO_SHADOWS, (0, 0, 0))
        self.render.scene.show_axes(False)

        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        return img

    def render_pose_trajectory_cv2(self,
                                   base_img_ref: np.ndarray,
                                   frame: int) -> np.ndarray:
        """ A OpenCV implementation of drawing frustums and trajectories 
        (line primitives)

        Args:
            frame_color_pairs: list of tuples. 
                [(frame_idx, color), ...]
                frame_idx: int. I.e. Frame number in reader's original video
                color: str. Color code
            traj_len: int. Number of trajectory lines to show
                30 frames = 0.5 sec in Epic
        
        Returns:
            img: np.ndarray RGB
        """
        out_img = base_img_ref.copy()
        num_images = self.reader.num_mp4_frames
        
        if frame < 0 or frame >= num_images:
            raise ValueError(f"Frame {frame} out of range [0, {num_images})")

        c2w = self.c2ws[frame]
        if c2w is None:
            raise ValueError(f"c2w is None for frame {frame}.")

        frustum_color = code_to_rgb('red')
        frustum = get_frustum(
            c2w=c2w, sz=self.frustum_size, to_geomtype=True)
        frustum_lines2d = self.cv2_render.project_polygon(
            frustum.points, frustum.lines)
        out_img = self.cv2_render.draw_lines(
            out_img, frustum_lines2d, color=frustum_color,
            thickness=self.frustum_line_width_cv2)

        return out_img

    def render_frame(self, 
                     frame: int,
                     ) -> np.ndarray:
        """ Draw multiple cameras with trajectories and frustums
            for a single frame.
        
        Args:
            frame_idx: int, used for caching colliding frames.
        """
        if self.base_scene_img is None:
            self.base_scene_img = self.render_base_scene()
        if self.cv2_render is None:
            self.cv2_render = CV2PrimitiveDrawer(
                render=self.render,
                width=self.out_size[0],
                height=self.out_size[1])
        out_img = self.render_pose_trajectory_cv2(
            base_img_ref=self.base_scene_img,
            frame=frame)
        return out_img

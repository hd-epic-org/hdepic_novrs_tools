from typing import List
import os
from pathlib import Path
import gzip
import json
import pickle
import pandas as pd
import numpy as np
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
import projectaria_tools.core.mps as mps
from projectaria_tools.core import calibration
from novrs_lib.construct_pinhole import get_pinhole_calibration
from novrs_lib.serialisation import calibration_from_dict
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("decord is not available. Please install it to read mp4 files.")


UNDEFINED_ROW = 'UNDEFINED_ROW'


class NoVRSReader:
    """
    """

    def __init__(self,
                 vid: str,
                 frame_type: str = 'mp4',
                 load_frame_traj=True,
                 load_mp4=False,
                 load_pts=False,
                 load_obs=False,
                 storage_dir='./hdepic_storage',
                 ):
        assert frame_type == 'mp4', 'Only mp4 frame supported'
        self.frame_type = frame_type
        self.storage_dir = storage_dir
        self.intermediate_dir = Path(storage_dir)/'intermediate_data'
        self.pid = vid.split('-')[0]
        self.vid = vid

        self._setup_mp4_csv()
        self._load_calibration(self.intermediate_dir)

        if load_mp4:
            mp4_path = Path(self.storage_dir)/f'Videos/{self.pid}/{self.vid}.mp4'
            mp4_path = str(mp4_path)
            self.mp4 = decord.VideoReader(mp4_path, ctx=decord.cpu(0))

        if load_frame_traj:
            self.frame_traj = self._load_frame_trajectory()

        if load_pts or load_obs:  # SLAM will be needed
            multi_dir = Path(storage_dir)/f'SLAM-and-Gaze/{self.pid}/SLAM/multi'
            # Get slam id
            with open(multi_dir/"vrs_to_multi_slam.json", 'r') as fp:
                vrs_to_slam = json.load(fp)
                slam_id = vrs_to_slam[f'{self.pid}/{vid}.vrs']
            self.slam_dir = multi_dir/f'{slam_id}/slam/'

            self.semi_pts_path = self.slam_dir/'semidense_points.csv.gz'
            if load_pts:
                with open(self.semi_pts_path, 'rb') as fp:
                    gzip_fp = gzip.GzipFile(fileobj=fp)
                    self.semi_pts = pd.read_csv(gzip_fp)  # csv
        
            if load_obs:
                obs_path = self.slam_dir/'semidense_observations.csv.gz'
                assert os.path.isfile(obs_path), f"Observation file {obs_path} not found."
                self.obs = pd.read_csv(obs_path)

        self.load_mp4 = load_mp4
        self.load_pts = load_pts
        self.load_frame_traj = load_frame_traj
        self.load_obs = load_obs

    @property
    def num_rgb_frames(self):
        return self.num_mp4_frames
    
    def _load_calibration(self, intermediate_dir: Path):
        rgb_camera_calibration = Path(
            intermediate_dir)/'rgb_camera_calibration'/f'{self.vid}.json'
        with open(rgb_camera_calibration) as fp:
            rgb_camera_calibration = json.load(fp)
        rgb_calib = calibration_from_dict(rgb_camera_calibration)
        self.rgb_camera_calibration = rgb_calib
        self.pinhole_params = get_pinhole_calibration(rgb_calib)
        self.pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(
            self.pinhole_params)

        self.T_rgb_to_slamL = \
            rgb_calib.get_transform_device_camera().to_matrix() # DEPRECATED. Transform from rgb to slam_l 
        self.T_pinholecw90_to_slamL = \
            self.pinhole_cw90.get_transform_device_camera().to_matrix() # Transform from rotated-pinhole to slam_l

    def _setup_mp4_csv(self):
        """ Setup time correspondence between Kranti's new_mp4 and vrs """
        # self.mp4_csv_path = Path(self.storage_dir)/f'new_mp4s/{self.pid}/{self.vid}_mp4_to_vrs_time_ns.csv'
        self.mp4_csv_path = Path(self.storage_dir)/f'Videos/{self.pid}/{self.vid}_mp4_to_vrs_time_ns.csv'
        self.mp4_to_vrs = pd.read_csv(self.mp4_csv_path)
        self.num_mp4_frames = len(self.mp4_to_vrs)

    def _load_frame_trajectory(self):
        self.frame_traj_path = \
            self.intermediate_dir/f'{self.frame_type}_trajectory'/f'{self.vid}.csv'
        assert self.frame_traj_path.exists(), "intermediate data is needed"
        frame_traj = mps.read_closed_loop_trajectory(str(self.frame_traj_path))
        # Make None pose explicit
        frame_traj = [
            pose_info if pose_info.graph_uid != UNDEFINED_ROW else None
            for pose_info in frame_traj
        ]
        return frame_traj
    
    def load_all_points(self, as_open3d=False) -> np.ndarray:
        """ returns: (N, 3) """
        assert self.load_pts, "Points not loaded"
        pts = np.asarray(self.semi_pts[['px_world', 'py_world', 'pz_world']])
        if as_open3d:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            return pcd
        return pts

    def load_filtered_points(self, 
                             threshold_invdep: float = 0.005,
                             threshold_dep: float = 0.01,
                             as_open3d=False) -> np.ndarray:
        """ returns: (N, 3) 
        The default thresholds are copied from projectaria_tools. 

        Args:
            threshold_invdep: float. Larger threshold means more noise included.
            threshold_dep: float. 
        """
        assert self.load_pts, "Points not loaded"
        INV_DIST_STD = 'inv_dist_std'
        DIST_STD = 'dist_std'
        sel_pts = self.semi_pts[
            (self.semi_pts[INV_DIST_STD] < threshold_invdep) & (self.semi_pts[DIST_STD] < threshold_dep)]
        pts = np.asarray(sel_pts[['px_world', 'py_world', 'pz_world']])
        if as_open3d:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            return pcd
        return pts

    def load_observations(self) -> pd.DataFrame:
        assert self.load_obs, "Observations not loaded"
        return self.obs

    def load_observations_at_timestamp(
            self,
            timestamp_ns: int,
            threshold: bool = True,
            threshold_invdep: float = 0.005,
            threshold_dep: float = 0.01,
            as_open3d=False
            ) -> np.ndarray:
        assert self.load_obs, "Observed points not loaded"
        assert self.obs['uid'].nunique() == self.semi_pts['uid'].nunique(), \
            '''
            Number of unique points in observations and semi_pts should be same. Maybe you're loading 
            observation points and semi-dense points from different recording sessions.
            '''
        selected_uids = self.obs[self.obs['frame_tracking_timestamp_us'] == timestamp_ns]['uid'].values
        selected_points = self.semi_pts[self.semi_pts['uid'].isin(selected_uids)]
        if threshold:
            selected_points = selected_points[
                (selected_points['inv_dist_std'] < threshold_invdep) & 
                (selected_points['dist_std'] < threshold_dep)]
        selected_points = np.asarray(selected_points[['px_world', 'py_world', 'pz_world']])
        if as_open3d:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected_points))
            return pcd
        return selected_points

    def load_slaml_trajectory(self, ret_mat=True) -> List:
        """ The slam_l poses at rgb_frames, in world coor.

        Args:
            ret_mat: bool. If True, return the 4x4 matrix.
                otherwise, return the Sophus SE3 object.

        Returns: 
            poses: list of (4, 4), slamL-to-world
                if pose at a frame is not available, pose=None.
        """
        assert self.load_frame_traj, "Frame trajectory not loaded"
        poses = [None for _ in range(self.num_rgb_frames)]
        for i, f in enumerate(range(self.num_rgb_frames)):
            pose_info = self.frame_traj[i]
            if pose_info is None:
                continue
            pose = pose_info.transform_world_device
            if ret_mat:
                pose = pose.to_matrix()
            poses[i] = pose
        return poses

    def load_rgb_trajectory(self, ret_mat=True) -> List:
        raise ValueError("Use load_pinholecw90_tracjectory() instead.")

    def load_pinholecw90_trajectory(self, ret_mat=True) -> List:
        """ Poses of pinhole camera ClockWise rotated 90 degrees 
            at rgb_frames
        """
        slaml_poses = self.load_slaml_trajectory(ret_mat=ret_mat)
        poses = []
        for pose in slaml_poses:
            if pose is None:
                poses.append(None)
            else:
                poses.append(pose @ self.T_pinholecw90_to_slamL)
        return poses
    
    def load_pinholecw90_trajectory_pickle(self, ret_mat=True) -> List:
        """ Optionally, load pinholecw90 poses from pickle file for faster loading.
        """
        assert self.load_frame_traj, "Frame trajectory not loaded"
        poses_path = Path(
            self.intermediate_dir)/f'pinholecw90_poses/{self.vid}.pkl'
        with open(poses_path, 'rb') as fp:
            return pickle.load(fp)

    def read_vrs_frame(self, *args) -> np.ndarray:
        raise ValueError("VRS Not supported. Use read_mp4_frame() instead.")

    def read_mp4_frame(self, 
                       frame: int,
                       undistort: bool,
                       ret_timestamp_ns=False,
                       no_warning=False):
        """ 
        Although this function is implemented, consider using decord's iterator
        for maximal efficiency, e.g.
            1) for frame_idx, img_arr in reader.mp4
            2) reader.mp4[100:200]

        Args:
            mp4_idx: frame index, start frame 0.
            mp4_sec: float, second start from 0s.
        """
        if no_warning == False:
            if not hasattr(self, '_warning_counter'):
                self._warning_counter = 0
            self._warning_counter += 1
            if self._warning_counter >= 20:
                print("Warning: consider using decord's iterator for max efficienty ")
        img_arr = self.mp4[frame].asnumpy()
        if undistort:
            img_arr = calibration.distort_by_calibration(
                img_arr, self.pinhole_params, self.rgb_camera_calibration)
        if ret_timestamp_ns:
            time_ns = int(self.mp4_to_vrs.iloc[frame]['vrs_device_time_ns'])
            return img_arr, time_ns
        return img_arr
    
    def read_mp4_by_sec(self, sec: float, undistort: bool, 
                        ret_timestamp_ns=False):
        raise NotImplementedError("Use read_mp4_frame() with frame index instead.")
    
    def undistort_image(self, img):
        """ Take an external image (e.g. mask, depth), do undistort"""
        img = calibration.distort_by_calibration(
            img, self.pinhole_params, self.rgb_camera_calibration)
        return img
    
    def get_mp4_timestamp(self, mp4_frame: int) -> int:
        """ Returns: timestamp_ns: int """
        time_ns = int(self.mp4_to_vrs.iloc[mp4_frame]['vrs_device_time_ns'])
        return time_ns
""" Aria VRS by default is fisheye, but we want to use pinhole model in slam_reader code. """

from projectaria_tools.core import calibration
from projectaria_tools.core.calibration import CameraCalibration


def get_pinhole_calibration(rgb_calib: CameraCalibration) -> CameraCalibration:
    """
    Returns:
        pinhole_params: CameraCalibration

        To get cw90 params and transformations, further do:
            pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(pinhole_params)
            T_pinholecw90_to_slamL = \
                pinhole_cw90.get_transform_device_camera().to_matrix() # Transform from rotated-pinhole to slam_l
    """
    PINHOLE_RARIO = 3.0139  # Sugguested by Ahmad
    img_w, img_h = rgb_calib.get_image_size()
    pinhole_params = calibration.get_linear_camera_calibration(
        img_w, img_h, focal_length=img_w / PINHOLE_RARIO,
        T_Device_Camera=rgb_calib.get_transform_device_camera())
    return pinhole_params
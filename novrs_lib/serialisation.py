""" Very dirty functions to convert aria's CameraCalibration to/from dict
This serialisation enables saving/loading aria's camera params.
We do this because aria's tool doesn't support exporting camera calibration stuffs in python
"""
import numpy as np
from projectaria_tools.core.calibration import CameraModelType, CameraCalibration
from projectaria_tools.core.sophus import SE3

def dict_from_calibration(calib: CameraCalibration) -> dict:
    """ export to dict to allow serialisation/dumping """
    projection_params = calib.projection_params().tolist()
    image_width, image_height = calib.get_image_size().tolist()
    mat3x4 = calib.get_transform_device_camera().to_matrix3x4().tolist()
    calib_params = {
        'label': calib.get_label(),                         # arg0
        'model_name.name': calib.model_name().name,         # arg1
        'projection_params': projection_params,             # arg2
        'T_Device_Camera.mat3x4': mat3x4,                   # arg3         'T_Device_Camera': SE3.from_matrix3x4(mat3x4),                # arg3
        'image_width': image_width,                         # arg4
        'image_height': image_height,                       # arg5
        'maybe_valid_radius': calib.get_valid_radius(),     # arg6
        'max_solid_angle': calib.get_max_solid_angle(),     # arg7
        'serial_number': calib.get_serial_number()          # arg8
    }
    return calib_params
    
def calibration_from_dict(calib_params: dict) -> CameraCalibration:
    """ This load from a pure dict 
    Returns:
        rgb_camera_calibration: CameraCalibration
    """
    model_name = getattr(CameraModelType, calib_params['model_name.name'])

    projection_params = np.asarray(calib_params['projection_params'], dtype=np.float64)

    mat3x4 = calib_params['T_Device_Camera.mat3x4']
    mat3x4 = np.asarray(mat3x4, dtype=np.float64).reshape(3, 4)
    T_Device_Camera = SE3.from_matrix3x4(mat3x4)
    
    args = (
        calib_params['label'],
        model_name,
        projection_params,
        T_Device_Camera,
        calib_params['image_width'],
        calib_params['image_height'],
        calib_params['maybe_valid_radius'],
        calib_params['max_solid_angle'],
        calib_params['serial_number']
    )
    return CameraCalibration(*args)
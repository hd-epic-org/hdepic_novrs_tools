from novrs_lib.novrs_reader_basic import NoVRSReader


# Example usage
vid = 'P01-20240202-161948'
frame = 100
reader = NoVRSReader(vid=vid, 
                     storage_dir='./hdepic_storage',)

# 1. Load the camera pose at a specified frame
cam_poses = reader.load_pinholecw90_trajectory()
cam_pose = cam_poses[frame]

# 2. Visualise the camera pose at this frame
from hovering_lib.hover_drawer import HoverDrawer
from PIL import Image
drawer = HoverDrawer(reader)
rend = drawer.render_frame(frame)
Image.fromarray(rend).save(f'{vid}_frame{frame:04d}_rendered.png')


# Additional functionalities:
# 3. Undistort images using calibration info stored in intermediate_data/ 
reader_with_mp4 = NoVRSReader(vid=vid, 
                        load_mp4=True,
                        storage_dir='./hdepic_storage',)
img = reader_with_mp4.read_mp4_frame(frame, undistort=False)
img_undistorted = reader_with_mp4.undistort_image(img)

# 4. Or do other things with reader.rgb_camera_calibration. (omitted)
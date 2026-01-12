import numpy as np
import cv2


class CV2PrimitiveDrawer:
    """ For points and lines, we don't need to use Open3D.
    OpenCV is enough.
    """
    def __init__(self, 
                 render,
                 width: int = 1280,
                 height: int = 720):
        """
        Args:
            render: Open3D renderer
            width: int, width of the image
            height: int, height of the image
        """
        self.width = width
        self.height = height
        self.fov_deg = render.scene.camera.get_field_of_view()
        self.K = self.get_intrinsic(self.fov_deg, width, height)
        self.R_cv, self.t_cv = self.get_extrinsic_from_o3d(render)
    
    def get_extrinsic_from_o3d(self, render):
        M = render.scene.camera.get_view_matrix()     # 4x4
        R = M[:3, :3]
        t = M[:3, 3]
        gl_to_cv = np.diag([1, 1, -1])  # flip Z axis
        R_cv = gl_to_cv @ R
        t_cv = gl_to_cv @ t
        return R_cv, t_cv

    def get_intrinsic(self, fov_deg, width, height):
        fov_rad = np.deg2rad(fov_deg)
        fy = height / (2 * np.tan(fov_rad / 2))
        fx = fy
        cx = width / 2
        cy = height / 2
        return np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]])

    def project_point(self, X_world):
        R = self.R_cv
        t = self.t_cv
        K = self.K
        X_cam = R @ X_world + t       # 3D in camera space
        x_proj = K @ X_cam
        x_proj /= x_proj[2]           # perspective divide
        x, _y = x_proj[:2]             # 2D pixel coordinates
        y = self.height - _y
        return int(x), int(y)
    
    def project_polygon(self, points, lines) -> list:
        """ Generalised function of get_frustum_lines2d.
        Use to project a polygon in 3D space to 2D for drawing.

        Args:
            points: np.ndarray, Nx3 array of 3D points
            lines: list of tuples, each tuple contains two indices of 
                points to form a line

        Returns:
            lines2d: list of tuples. [(x1, y1), (x2, y2)]
        """
        lines2d = []
        for line in lines:
            p1 = points[line[0]]
            p2 = points[line[1]]
            x1, y1 = self.project_point(p1)
            x2, y2 = self.project_point(p2)
            lines2d.append(((x1, y1), (x2, y2)))
        return lines2d
    
    def get_trajectory_line2d(self, 
                              pos_history) -> list:
        """
        Args:
            pos_history: list of 3D points
            num_line: int, number of lines to draw
            line_radius: float, radius of the line
        Returns:
            lines2d: list of tuples. [(x1, y1), (x2, y2)]
        """
        lines2d = []
        for i in range(len(pos_history) - 1):
            p1 = pos_history[i]
            p2 = pos_history[i + 1]
            x1, y1 = self.project_point(p1)
            x2, y2 = self.project_point(p2)
            lines2d.append(((x1, y1), (x2, y2)))
        
        return lines2d
    
    def draw_lines(self, img, lines2d, color=(0, 255, 0), thickness=2):
        """
        Draw lines on the image
        Args:
            img: np.ndarray, image to draw on
            lines2d: list of tuples. [(x1, y1), (x2, y2)]
            color: tuple, color of the line
            thickness: int, thickness of the line
        Returns:
            img: np.ndarray, image with lines drawn
        """
        for line in lines2d:
            cv2.line(img, line[0], line[1], color, thickness,
                    lineType=cv2.LINE_AA)
        return img
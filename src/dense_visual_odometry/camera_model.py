import numpy as np
from pathlib import Path
import logging
import yaml
from typing import Union

from dense_visual_odometry.utils.lie_algebra import SE3
from dense_visual_odometry.utils.numpy_cache import np_cache


logger = logging.getLogger(__name__)


# TODO: Support distorssion coefficients
class RGBDCameraModel:
    """
        Class that models a camera using the pinhole model
    """

    # Keywords for loading camera model from a config file
    INTRINSICS_KEYWORD = "intrinsics"
    DEPTH_SCALE_KEYWORD = "depth_scale"
    DISTORSSION_COEFFS_KEYWORD = "distorssion_coefficients"
    DISTORSSION_MODEL_KEYWORD = "distorssion_model"

    def __init__(
        self, intrinsics: np.ndarray, depth_scale: float, distorssion_coeffs: Union[np.ndarray, None] = None,
        distorssion_model: Union[str, None] = None
    ):
        """
            Creates a RGBDCameraModel instance

        Parameters
        ----------
        intrinsics : np.ndarray
            3x3 instrinsics matrix
                [fx  s cx]
                [ 0 fy cy]
                [ 0  0  1]
            with:
                fx, fy: Focal lengths for the sensor in X and Y dimensions
                s: Any possible skew between the sensor axes caused by sensor not being perpendicular from optical axis
                cx, cy: Image center expressed in pixel coordinates
        depth_scale : float
            Scale (multiplication factor) used to convert from the sensor Digital Number to meters
        """
        assertion_message = f"Expected a 3x3 'intrinsics', got {intrinsics.shape} instead"
        assert intrinsics.shape == (3, 3), assertion_message
        assertion_message = "Expected 'scale' to be a positive floating point, got '{:.3f}' instead".format(depth_scale)
        assert depth_scale >= 0.0, assertion_message

        # Store calibration matrix as a 3x4 matrix
        self._intrinsics = np.zeros((3, 4), dtype=np.float32)
        self._intrinsics[:3, :3] = intrinsics
        self.depth_scale = depth_scale

        self.distorssion_coeffs = distorssion_coeffs
        self.distorsion_model = distorssion_model

    @classmethod
    def load_from_yaml(cls, filepath: Path):
        """
            Loads RGBDCameraModel instance from a YAML file

        Parameters
        ----------
        filepath : Path
            Path to configuration file

        Returns
        -------
        camera_model : RGBDCameraModel | None
            Camera model loaded from file if possible, None otherwise
        """

        if not filepath.exists():
            logger.error("Could not find configuration file '{}'".format(str(filepath)))
            return None

        with filepath.open("r") as fp:
            data = yaml.load(fp, yaml.Loader)

        try:
            camera_matrix = np.array(data[cls.INTRINSICS_KEYWORD], dtype=np.float32)
            depth_scale = data[cls.DEPTH_SCALE_KEYWORD]
            distorssion_coefficients = data.get(cls.DISTORSSION_COEFFS_KEYWORD)
            distorssion_model = data.get(cls.DISTORSSION_MODEL_KEYWORD)

        except KeyError as e:
            logger.error(e)
            return None

        return cls(camera_matrix, depth_scale, distorssion_coefficients, distorssion_model)

    # NOTE: Does compute invalid depth, user must do it
    def deproject_unsafe(
        self, depth_image: np.ndarray, camera_pose: np.ndarray, mask: np.ndarray, level: int = 0
    ):
        """
            Deprojects a depth image into the World reference frame without taking into account no valid
            pixels (user must defined mask of valid pixels)

        Parameters
        ----------
        depth_image : np.ndarray
            Depth image (with invalid pixels defined with the value 0)
        camera_pose : np.ndarray
            Camera pose w.r.t World coordinate frame expressed as a 6x1 se(3) vector

        Returns
        -------
        pointcloud : np.ndarray
            3D Point (4xN) coordinates of projected points in Homogeneous coordinates (i.e x, y, z, 1)
        """
        assertion_msg = "Expected 'depth_image' and 'mask' to have the same shape, got '{}' and '{}'".format(
            depth_image.shape, mask.shape
        )
        assert depth_image.shape == mask.shape, assertion_msg

        height, width = depth_image.shape

        z = (depth_image[mask] * self.depth_scale).astype(np.float32)

        # Compute sensor grid
        # TODO: Use a more efficient way of creating pointcloud -> Several pixels values are repeated. See `sparse`
        # parameter of `np.meshgrid`
        x_pixel, y_pixel = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), copy=False
        )
        x_pixel = x_pixel[mask]
        y_pixel = y_pixel[mask]

        scale_matrix = np.array([
            [2 ** level, 0, 2 ** (level - 1) - 0.5],
            [0, 2 ** level, 2 ** (level - 1) - 0.5],
            [0, 0, 1]
        ], dtype=np.float32)

        # Map from pixel position to 3d coordinates using camera matrix (inverted)
        # Get x, y points w.r.t camera reference frame (still not multiply by the depth)
        points = np.dot(
            np.linalg.inv(np.dot(scale_matrix, self._intrinsics[:3, :3])),
            np.vstack((x_pixel, y_pixel, np.ones_like(z)))
        )

        pointcloud = np.vstack((points[0, :] * z, points[1, :] * z, z, np.ones_like(z)))

        # Convert from camera reference frame to world reference frame
        pointcloud = np.dot(SE3.exp(camera_pose), pointcloud)

        return pointcloud

    @np_cache
    def deproject(
        self, depth_image: np.ndarray, camera_pose: np.ndarray, return_mask: bool = False, level: int = 0,
        mask_pixels: np.ndarray = None
    ):
        """
            Deprojects a depth image into the World reference frame

        Parameters
        ----------
        depth_image : np.ndarray
            Depth image (with invalid pixels defined with the value 0)
        camera_pose : np.ndarray
            Camera pose w.r.t World coordinate frame expressed as a 6x1 se(3) vector
        return_mask : bool
            if True, then a bolean mask is returned with valid pixels

        Returns
        -------
        pointcloud : np.ndarray
            3D Point (4xN) coordinates of projected points in Homogeneous coordinates (i.e x, y, z, 1)
        mask : np.ndarray, optional
            Boolean mask with the same shape as `depth_image` with True on valid pixels and false on non valid.
        """
        height, width = depth_image.shape

        # Remove invalid points
        mask = depth_image != 0.0
        if mask_pixels is not None:
            mask = np.logical_and(mask, mask_pixels)
        mask = mask.reshape(-1)

        z = depth_image.reshape(-1) * self.depth_scale
        z = z[mask].astype(np.float32)

        # Compute sensor grid
        # TODO: Use a more efficient way of creating pointcloud -> Several pixels values are repeated. See `sparse`
        # parameter of `np.meshgrid`
        x_pixel, y_pixel = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), copy=False
        )
        x_pixel = x_pixel.reshape(-1)
        y_pixel = y_pixel.reshape(-1)

        x_pixel = x_pixel[mask]
        y_pixel = y_pixel[mask]

        scale_matrix = np.array([
            [2 ** level, 0, 2 ** (level - 1) - 0.5],
            [0, 2 ** level, 2 ** (level - 1) - 0.5],
            [0, 0, 1]
        ], dtype=np.float32)

        # Map from pixel position to 3d coordinates using camera matrix (inverted)
        # Get x, y points w.r.t camera reference frame (still not multiply by the depth)
        points = np.dot(
            np.linalg.inv(np.dot(scale_matrix, self._intrinsics[:3, :3])),
            np.vstack((x_pixel, y_pixel, np.ones_like(z)))
        )

        pointcloud = np.vstack((points[0, :] * z, points[1, :] * z, z, np.ones_like(z)))

        # Convert from camera reference frame to world reference frame
        pointcloud = np.dot(SE3.exp(camera_pose), pointcloud)

        if return_mask:
            return (pointcloud, mask.reshape(depth_image.shape))

        return pointcloud

    @np_cache
    def project(self, pointcloud: np.ndarray, camera_pose: np.ndarray, level: int = 0):
        """
            Projects given pointcloud to image plane

        Parameters
        ----------
        pointcloud : np.ndarray
            3D Point (4xN) coordinates of projected points in Homogeneous coordinates (i.e x, y, z, 1) w.r.t the World
            coordinate frame
        camera_pose : np.ndarray
            Camera pose w.r.t World reference frame expressed as a 6x1 se(3) vector

        Returns
        -------
        points_pixel : np.ndarray
            Image plane (3xN) coordinates given in homogeneous coordinates (i.e [u, v, 1]) in pixels. Note that
            values might not be integer and might lie between physical pixels, user must then decide what to do
            with those
        """
        scale_matrix = np.array([
            [2 ** level, 0, 2 ** (level - 1) - 0.5],
            [0, 2 ** level, 2 ** (level - 1) - 0.5],
            [0, 0, 1]
        ], dtype=np.float32)

        camera_matrix = np.dot(np.dot(scale_matrix, self._intrinsics), SE3.inverse(SE3.exp(camera_pose)))
        points_pixels = np.dot(camera_matrix, pointcloud)
        points_pixels /= points_pixels[2, :]

        return points_pixels

    @property
    def intrinsics(self):
        return self._intrinsics

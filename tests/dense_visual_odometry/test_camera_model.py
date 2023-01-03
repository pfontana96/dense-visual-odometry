from unittest.mock import MagicMock, Mock, patch
import pytest

import numpy as np
import yaml

from dense_visual_odometry.camera_model import RGBDCameraModel
# from dense_visual_odometry.utils.lie_algebra import Se3, So3


class TestRGBDCameraModel:

    def test_given_calib_matrix_and_scale__when_init__then_ok(self):

        # Given
        calib_matrix = np.eye(3, dtype=np.float32)
        scale = 1.0

        result = RGBDCameraModel(calib_matrix, scale, 10, 10)

        # Then
        assert result is not None
        expected_calib_matrix = np.zeros((3, 4), dtype=np.float32)
        expected_calib_matrix[:3, :3] = calib_matrix
        np.testing.assert_equal(result.intrinsics, expected_calib_matrix)
        assert result.depth_scale == scale

    def test__given_not_valid_calib_matrix__when_init__then_raise_assertion(self):

        # Given
        calib_matrix = np.random.rand(5, 5)
        scale = 1.0

        # When + Then
        with pytest.raises(AssertionError):
            _ = RGBDCameraModel(calib_matrix, scale, 10, 10)

    def test__given_not_valid_depth_scale__when_init__then_raises_assertion(self):

        # Given
        calib_matrix = np.eye(3)
        scale = -1.0

        # When + Then
        with pytest.raises(AssertionError):
            _ = RGBDCameraModel(calib_matrix, scale, 10, 10)

    def test__given_valid_yaml_file__when_load_from_yaml__then_ok(self):

        # Given
        valid_file_content = yaml.dump({
            RGBDCameraModel.INTRINSICS_KEYWORD: np.eye(3, dtype=np.float32),
            RGBDCameraModel.DEPTH_SCALE_KEYWORD: 1.0,
        })

        # Mock pathlib.Path
        path_mock = MagicMock()  # MagicMock allows to override __enter__ for context managers
        path_mock.exists.return_value = True
        path_mock.open.return_value = path_mock
        path_mock.__enter__.return_value = valid_file_content

        # When
        result = RGBDCameraModel.load_from_yaml(path_mock)

        # Then
        path_mock.exists.assert_called_once()
        path_mock.open.assert_called_once_with("r")

        expected_intrinsics = np.zeros((3, 4), dtype=np.float32)
        expected_intrinsics[:3, :3] = np.eye(3, dtype=np.float32)
        np.testing.assert_equal(result.intrinsics, expected_intrinsics)
        assert result.depth_scale == 1.0

    @patch("dense_visual_odometry.camera_model.logger")
    def test__given_not_valid_yaml_file__when_load_from_yaml__then_none(self, logger_mock):

        # Given
        not_valid_file_content = yaml.dump({
            "not": "valid",
            "yaml": "file"
        })

        # Mock pathlib.Path
        path_mock = MagicMock()  # MagicMock allows to override __enter__ for context managers
        path_mock.exists.return_value = True
        path_mock.open.return_value = path_mock
        path_mock.__enter__.return_value = not_valid_file_content

        # When
        result = RGBDCameraModel.load_from_yaml(path_mock)

        # Then
        assert result is None
        logger_mock.error.assert_called_once()

    @patch("dense_visual_odometry.camera_model.logger")
    def test__given_non_existent_file__when_load_from_yaml__then_none(self, logger_mock):
        # Given
        path_mock = Mock()
        path_mock.exists.return_value = False

        # When
        result = RGBDCameraModel.load_from_yaml(path_mock)

        # Then
        path_mock.exists.assert_called_once()
        assert result is None
        logger_mock.error.assert_called_once()

    def test__given_known_depth_image__when_deproject__then_ok(self):

        # Given
        height, width = (10, 10)
        depth_image = np.ones((height, width), dtype=np.float32)
        depth_image[:5] = 2.0
        depth_image[5:, :2] = 3.0
        depth_scale = 0.5

        # Perfect pin-hole camera with 0.5 scale
        camera_model = RGBDCameraModel(intrinsics=np.eye(3, dtype=np.float32), depth_scale=depth_scale)

        # When
        pointcloud = camera_model.deproject(depth_image)

        # Then
        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        z = depth_image.reshape(1, -1) * depth_scale
        expected_pointcloud = np.vstack(
            (x.reshape(1, -1) * z, y.reshape(1, -1) * z, z, np.ones((1, height * width), dtype=np.float32))
        )
        np.testing.assert_equal(pointcloud, expected_pointcloud)

    def test__depth_given_image_with_invalid_pixels_and_return_mask__when_deproject__then_ok(self):
        # Given
        height, width = (10, 10)
        depth_image = np.ones((height, width), dtype=np.float32)
        depth_image[:5] = 2.0
        depth_image[5:, :2] = 3.0
        depth_image[0, 0] = 0.0
        depth_image[0, 1] = 0.0
        depth_scale = 0.5

        camera_model = RGBDCameraModel(intrinsics=np.eye(3, dtype=np.float32), depth_scale=depth_scale)

        # When
        pointcloud, mask = camera_model.deproject(depth_image, return_mask=True)

        # Then
        expected_mask = (depth_image != 0.0)
        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        x = x[expected_mask]
        y = y[expected_mask]
        z = depth_image[expected_mask].reshape(1, -1) * depth_scale
        expected_pointcloud = np.vstack(
            (x.reshape(1, -1) * z, y.reshape(1, -1) * z, z, np.ones_like(x).reshape(1, -1))
        )

        np.testing.assert_equal(pointcloud, expected_pointcloud)
        np.testing.assert_equal(mask, expected_mask)

    def test__given_a_known_pointcloud__when_project__then_ok(self):

        # Given
        height, width = (5, 5)
        depth_scale = 0.5

        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        z = np.arange(height * width).reshape(1, -1)
        pointcloud = np.vstack(
            (x.reshape(1, -1), y.reshape(1, -1), z, np.ones((1, height * width), dtype=np.float32))
        )

        camera_model = RGBDCameraModel(intrinsics=np.eye(3, dtype=np.float32), depth_scale=depth_scale)

        # When
        pixel_points = camera_model.project(pointcloud)[:2]

        # Then
        expected_pixel_points = pointcloud[:2] / pointcloud[2]
        np.testing.assert_equal(pixel_points, expected_pixel_points)

    # TODO: Fix test using scipy interpolator
    # @pytest.mark.parametrize("load_single_benchmark_case", list(range(1, 10)), indirect=True)
    # def test__given_gray_and_depth_image__when_deproject_and_project__then_almost_equal(
    #     self, load_camera_intrinsics_file, load_single_benchmark_case
    # ):

    #     # Given
    #     camera_model = RGBDCameraModel.load_from_yaml(load_camera_intrinsics_file)

    #     gray_images, depth_images, transforms = load_single_benchmark_case
    #     gray_image = gray_images[0]
    #     depth_image = depth_images[0]

    #     camera_pose = Se3(So3(transforms[0][:3, :3]), transforms[0][:3, 3].reshape(3, 1))

    #     # When
    #     pointcloud, mask = camera_model.deproject(depth_image=depth_image, return_mask=True)
    #     pointcloud = np.dot(camera_pose.exp(), pointcloud)
    #     projected_points = camera_model.project(pointcloud=pointcloud)

    #     # Then
    #     projected_image = np.zeros_like(gray_image)
    #     projected_image[mask] = Interp2D.bilinear(
    #         x=projected_points[0], y=projected_points[1], image=gray_image, cast=True
    #     )

    #     # NOTE: By the current implementation (17/04/2022) of 'Interp2D.bilinear' if we give the exact
    #     # grid to retrieve the same image, then last row and last column will be 0.0
    #     np.testing.assert_allclose(
    #         gray_image[:-1, :-1][mask[:-1, :-1]], projected_image[:-1, :-1][mask[:-1, :-1]], atol=1.0
    #     )

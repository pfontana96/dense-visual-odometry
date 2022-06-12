from unittest import TestCase

import numpy as np

from dense_visual_odometry.core import DenseVisualOdometry
from dense_visual_odometry.camera_model import RGBDCameraModel


class TestDVO(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_keep_dims_false__when_compute_residuals__then_shape_is_correct(self):

        # Given
        calib_matrix = np.eye(3, dtype=np.float32)
        depth_scale = 1.0

        camera_model = RGBDCameraModel(calib_matrix, depth_scale)

        gray_image = np.zeros((5, 5), dtype=np.float32)

        depth_image = np.ones_like(gray_image)
        depth_image[1, 1] = 0.0  # Adds invalid depth pixel

        # When
        dvo = DenseVisualOdometry(camera_model, np.zeros((6, 1), dtype=np.float32))
        residuals = dvo.compute_residuals(gray_image, gray_image, depth_image, np.zeros((6, 1), dtype=np.float32),
                                          keep_dims=False, return_mask=True)  # return_mask should be ignored

        # Then
        self.assertIsInstance(residuals, np.ndarray)
        self.assertTupleEqual(residuals.shape, depth_image[depth_image != 0.0].reshape(-1, 1).shape)

    def test__given_return_mask_true__when_compute_residuals__then_ok(self):

        # Given
        calib_matrix = np.eye(3, dtype=np.float32)
        depth_scale = 1.0

        camera_model = RGBDCameraModel(calib_matrix, depth_scale)

        gray_image = np.zeros((5, 5), dtype=np.float32)

        depth_image = np.ones_like(gray_image)
        depth_image[1, 1] = 0.0  # Adds invalid depth pixel

        # When
        dvo = DenseVisualOdometry(camera_model, np.zeros((6, 1), dtype=np.float32))
        result = dvo.compute_residuals(gray_image, gray_image, depth_image, np.zeros((6, 1), dtype=np.float32),
                                       keep_dims=True, return_mask=True)  # return_mask should NOT be ignored

        # Then
        self.assertIsInstance(result, tuple)
        np.testing.assert_array_equal(result[1], depth_image != 0.0)

    def test__given_same_image_and_no_transform__when_compute_residuals__then_zero(self):

        # Given
        calib_matrix = np.eye(3, dtype=np.float32)
        depth_scale = 1.0

        camera_model = RGBDCameraModel(calib_matrix, depth_scale)

        height, width = (8, 15)
        intensity_value = 150
        gray_image = np.full(shape=(height, width), fill_value=intensity_value, dtype=np.uint8)
        gray_image[:int(height / 2), :int(width / 2)] = intensity_value / 3.0

        depth_image = np.ones_like(gray_image)
        depth_image[:int(height / 2), :int(width / 2)] = 3.0

        # When
        dvo = DenseVisualOdometry(camera_model, np.zeros((6, 1), dtype=np.float32))
        residuals = dvo.compute_residuals(gray_image, gray_image, depth_image, np.zeros((6, 1), dtype=np.float32))

        # Then
        # NOTE: By the current implementation (17/04/2022) of 'Interp2D.bilinear' if we give the exact grid to retrieve
        # the same image, then last row and last column will be 0.0
        np.testing.assert_almost_equal(residuals[:-1, :-1], np.zeros_like(gray_image[:-1, :-1], dtype=np.float32))

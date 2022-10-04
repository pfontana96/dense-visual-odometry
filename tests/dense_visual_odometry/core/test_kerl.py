import pytest

import numpy as np

from dense_visual_odometry.core import KerlDVO
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.utils.lie_algebra.special_euclidean_group import SE3


_PERFECT_IMAGE_SHAPE = (10, 10)


class TestDVO:

    @property
    def perfect_camera_model(self):
        return RGBDCameraModel(np.eye(3, dtype=np.float32), 1.0, _PERFECT_IMAGE_SHAPE[0], _PERFECT_IMAGE_SHAPE[1])

    def test__given_keep_dims_false__when_compute_residuals__then_shape_is_correct(self):

        # Given
        gray_image = np.zeros(_PERFECT_IMAGE_SHAPE, dtype=np.float32)

        depth_image = np.ones_like(gray_image)
        depth_image[1, 1] = 0.0  # Adds invalid depth pixel

        # When
        dvo = KerlDVO(self.perfect_camera_model, np.zeros((6, 1), dtype=np.float32), 1)
        residuals = dvo._compute_residuals(
            gray_image, gray_image, depth_image, np.zeros((6, 1), dtype=np.float32), keep_dims=False, return_mask=True
        )  # return_mask should be ignored

        # Then
        assert isinstance(residuals, np.ndarray)
        assert residuals.shape == depth_image[depth_image != 0.0].reshape(-1, 1).shape

    def test__given_return_mask_true__when_compute_residuals__then_ok(self):

        # Given
        gray_image = np.zeros(_PERFECT_IMAGE_SHAPE, dtype=np.float32)

        depth_image = np.ones_like(gray_image)
        depth_image[1, 1] = 0.0  # Adds invalid depth pixel

        # When
        dvo = KerlDVO(self.perfect_camera_model, np.zeros((6, 1), dtype=np.float32), 1)
        result = dvo._compute_residuals(
            gray_image, gray_image, depth_image, np.zeros((6, 1), dtype=np.float32), keep_dims=True, return_mask=True
        )  # return_mask should NOT be ignored

        # Then
        assert isinstance(result, tuple)
        np.testing.assert_array_equal(result[1], depth_image != 0.0)

    def test__given_same_image_and_no_transform__when_compute_residuals__then_zero(self):

        # Given
        height, width = _PERFECT_IMAGE_SHAPE
        intensity_value = 150
        gray_image = np.full(shape=(height, width), fill_value=intensity_value, dtype=np.uint8)
        gray_image[:int(height / 2), :int(width / 2)] = intensity_value / 3.0

        depth_image = np.ones_like(gray_image)
        depth_image[:int(height / 2), :int(width / 2)] = 3.0

        # When
        dvo = KerlDVO(self.perfect_camera_model, np.zeros((6, 1), dtype=np.float32), 1)
        residuals = dvo._compute_residuals(gray_image, gray_image, depth_image, np.zeros((6, 1), dtype=np.float32))

        # Then
        # NOTE: By the current implementation (17/04/2022) of 'Interp2D.bilinear' if we give the exact grid to retrieve
        # the same image, then last row and last column will be 0.0
        np.testing.assert_almost_equal(residuals[:-1, :-1], np.zeros_like(gray_image[:-1, :-1], dtype=np.float32))

    @pytest.mark.parametrize("load_single_benchmark_case", [5], indirect=True)
    def test__given_single_benchmark__when_find_optimal_transformation__then_ok(
        self, load_single_benchmark_case, load_camera_intrinsics_file
    ):

        # Given
        gray_images, depth_images, transformations = load_single_benchmark_case
        camera_model = RGBDCameraModel.load_from_yaml(load_camera_intrinsics_file)

        dvo = KerlDVO(camera_model, np.zeros((6, 1), dtype=np.float32), 1)

        # when
        result = dvo._find_optimal_transformation(
            gray_image=gray_images[1], gray_image_prev=gray_images[0], depth_image_prev=depth_images[0]
        )

        # Then
        expected_result = SE3.log(transformations[1]) - SE3.log(transformations[0])
        np.testing.assert_allclose(result, expected_result)

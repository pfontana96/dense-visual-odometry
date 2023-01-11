import pytest

import numpy as np

from dense_visual_odometry.core import KerlDVO
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.utils.lie_algebra.special_euclidean_group import Se3, So3


_PERFECT_IMAGE_SHAPE = (10, 10)


class TestDVO:

    @property
    def perfect_camera_model(self):
        return RGBDCameraModel(np.eye(3, dtype=np.float32), 1.0, _PERFECT_IMAGE_SHAPE[0], _PERFECT_IMAGE_SHAPE[1])

    def test__given_same_image_and_no_transform__when_compute_residuals__then_zero(self):

        # Given
        height, width = _PERFECT_IMAGE_SHAPE
        intensity_value = 150
        gray_image = np.full(shape=(height, width), fill_value=intensity_value, dtype=np.uint8)
        gray_image[:int(height / 2), :int(width / 2)] = intensity_value / 3.0

        depth_image = np.ones_like(gray_image)
        depth_image[:int(height / 2), :int(width / 2)] = 3.0

        # When
        dvo = KerlDVO(self.perfect_camera_model, Se3.identity(), 1)
        dvo._init_gray_image_interpolator(gray_image)
        dvo._init_gradients_interpolators(gray_image)
        residuals, _ = dvo._compute_residuals(gray_image, gray_image, depth_image, estimate=Se3.identity())

        # Then
        np.testing.assert_almost_equal(residuals, np.zeros_like(gray_image.reshape(-1, 1), dtype=np.float32))

    @pytest.mark.parametrize("load_single_benchmark_case", [4, 5], indirect=True)
    def test__given_images_with_ground_truth_estimate__when_compute_residuals__then_zero(
        self, load_single_benchmark_case, load_camera_intrinsics_file
    ):

        # Given
        gray_images, depth_images, transformations = load_single_benchmark_case
        camera_model = RGBDCameraModel.load_from_yaml(load_camera_intrinsics_file)

        gt_transform = Se3(
            So3(transformations[1][:3, :3]), transformations[1][:3, 3].reshape(3, 1)
            ).inverse() * Se3(
                So3(transformations[0][:3, :3]), transformations[0][:3, 3].reshape(3, 1)
            )

        # When
        dvo = KerlDVO(camera_model, transformations[0], 1)
        dvo._init_gray_image_interpolator(gray_images[1])
        dvo._init_gradients_interpolators(gray_images[1])
        residuals, _ = dvo._compute_residuals(gray_images[1], gray_images[0], depth_images[0], estimate=gt_transform)

        mask = depth_images[0] != 0
        res_img = np.zeros_like(gray_images[0], dtype=np.uint8)
        res_img[mask] = residuals.reshape(-1).astype(np.uint8)

        # Then
        assert np.isclose(residuals.mean(), 0.0, atol=5.0)
        assert residuals.std() < 20

    # @pytest.mark.parametrize("load_single_benchmark_case", [5], indirect=True)
    # def test__given_single_benchmark__when_find_optimal_transformation__then_ok(
    #     self, load_single_benchmark_case, load_camera_intrinsics_file
    # ):

    #     # Given
    #     gray_images, depth_images, transformations = load_single_benchmark_case
    #     camera_model = RGBDCameraModel.load_from_yaml(load_camera_intrinsics_file)

    #     dvo = KerlDVO(camera_model, transformations[0], 4)

    #     # when
    #     result = dvo._find_optimal_transformation(
    #         gray_image=gray_images[1], gray_image_prev=gray_images[0], depth_image_prev=depth_images[0]
    #     )

    #     # Then
    #     expected_result = transformations[1].inverse() * transformations[0]
    #     assert(result == expected_result)

import pytest

import numpy as np
import cv2

from dense_visual_odometry.core import get_dvo
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

        color_image = np.full(shape=(height, width, 3), fill_value=intensity_value, dtype=np.uint8)
        color_image[:int(height / 2), :int(width / 2), :] = intensity_value / 3.0

        depth_image = np.ones_like(gray_image)
        depth_image[:int(height / 2), :int(width / 2)] = 3.0

        # When
        dvo = get_dvo("robust-dvo", self.perfect_camera_model, Se3.identity(), **{"levels": 1})
        dvo.step(color_image=color_image, depth_image=depth_image)
        dvo._build_pyramids(
            gray_image=gray_image,
            depth_image=depth_image)
        dvo._setup(level=0)
        residuals, _, _ = dvo.compute_residuals_and_jacobian(estimate=Se3.identity(), level=0)

        # Then
        np.testing.assert_almost_equal(residuals, np.zeros_like(gray_image.reshape(-1, 1), dtype=np.float32))

    @pytest.mark.parametrize("load_single_benchmark_case", [4, 5], indirect=True)
    def test__given_images_with_ground_truth_estimate__when_compute_residuals__then_zero(
        self, load_single_benchmark_case, load_camera_intrinsics_file
    ):

        # Given
        color_images, depth_images, transformations = load_single_benchmark_case
        camera_model = RGBDCameraModel.load_from_yaml(load_camera_intrinsics_file)

        gt_transform = Se3(
            So3(transformations[1][:3, :3]), transformations[1][:3, 3].reshape(3, 1)
        ).inverse() * Se3(
            So3(transformations[0][:3, :3]), transformations[0][:3, 3].reshape(3, 1)
        )

        # When
        init_pose = Se3(So3(transformations[0][:3, :3]), transformations[0][:3, 3].reshape(3, 1))
        dvo = get_dvo("robust-dvo", camera_model, init_pose, **{"levels": 1})
        dvo.step(color_image=color_images[0], depth_image=depth_images[0])
        dvo._build_pyramids(
            gray_image=cv2.cvtColor(color_images[1], cv2.COLOR_BGR2GRAY),
            depth_image=depth_images[1])
        dvo._setup(level=0)
        residuals, _, _ = dvo.compute_residuals_and_jacobian(estimate=gt_transform, level=0)

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

import pytest

import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence

from dense_visual_odometry.utils.transform import find_rigid_body_transform_from_pointclouds_SVD, EstimationError


class TestFindRigidBodyTransformFromPointclouds:

    @pytest.fixture(scope="class")
    def pointcloud(self):
        rs = RandomState(MT19937(SeedSequence(0)))
        return (rs.rand(3, 30) - 0.5) * 5 + 3

    def test__given_same_pointcloud__then_return_identity(self, pointcloud):
        # Given + When
        T = find_rigid_body_transform_from_pointclouds_SVD(src_pc=pointcloud, dst_pc=pointcloud)

        # Then
        np.testing.assert_allclose(T, np.eye(4), atol=1e-6)

    def test__given_pointcloud_and_known_transform__then_return_transform(self, pointcloud):
        # Given
        transform = np.array([
            [1, 0, 0, 2],
            [0, 0, -1, 0],
            [0, 1, 0, -5],
            [0, 0, 0, 1]
        ])

        dst_pointcloud = np.dot(transform, np.vstack((pointcloud, np.ones((1, pointcloud.shape[1])))))

        # When
        T = find_rigid_body_transform_from_pointclouds_SVD(src_pc=pointcloud, dst_pc=dst_pointcloud[:3, :])

        # Then
        np.testing.assert_allclose(T, transform, atol=1e-6)

    def test_given_pointcloud_and_reflection__then_raises_estimationerror(self, pointcloud):
        # Given
        # Given
        transform = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        dst_pointcloud = np.dot(transform, np.vstack((pointcloud, np.ones((1, pointcloud.shape[1])))))

        # When + Then
        with pytest.raises(EstimationError):
            find_rigid_body_transform_from_pointclouds_SVD(src_pc=pointcloud, dst_pc=dst_pointcloud[:3, :])

    def test_given_noisy_pointclouds__then_ok(self, pointcloud):

        # Given
        transform = np.array([
            [1, 0, 0, 2],
            [0, 0, -1, 0],
            [0, 1, 0, -5],
            [0, 0, 0, 1]
        ])

        dst_pointcloud = np.dot(transform, np.vstack((pointcloud, np.ones((1, pointcloud.shape[1])))))
        dst_pointcloud += np.random.normal(loc=0.0, scale=0.15, size=dst_pointcloud.shape)  # Gaussian noise up to 30cm

        # When
        T = find_rigid_body_transform_from_pointclouds_SVD(src_pc=pointcloud, dst_pc=dst_pointcloud[:3, :])

        # Then
        np.testing.assert_allclose(T, transform, atol=0.15)

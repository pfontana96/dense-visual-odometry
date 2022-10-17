import pytest

import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence

from dense_visual_odometry.utils.transform import find_rigid_body_transform_from_pointclouds, EstimationError


class TestFindRigidBodyTransformFromPointclouds:

    @pytest.fixture(scope="class")
    def pointcloud(self):
        rs = RandomState(MT19937(SeedSequence(0)))
        return (rs.rand(3, 30) - 0.5) * 5 + 3

    def test__given_same_pointcloud__then_return_identity(self, pointcloud):
        # Given + When
        R, t, _ = find_rigid_body_transform_from_pointclouds(src_pc=pointcloud, dst_pc=pointcloud)

        # Then
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(t, np.zeros((3, 1)), atol=1e-6)

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
        R, t, _ = find_rigid_body_transform_from_pointclouds(src_pc=pointcloud, dst_pc=dst_pointcloud[:3, :])

        # Then
        np.testing.assert_allclose(R, transform[:3, :3], atol=1e-6)
        np.testing.assert_allclose(t, transform[:3, 3].reshape(3, 1), atol=1e-6)

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
            find_rigid_body_transform_from_pointclouds(src_pc=pointcloud, dst_pc=dst_pointcloud[:3, :])

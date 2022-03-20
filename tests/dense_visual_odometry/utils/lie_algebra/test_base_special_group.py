import numpy as np

from dense_visual_odometry.utils.lie_algebra.base_special_group import BaseSpecialGroup

from unittest import TestCase
from unittest.mock import patch


class TestBaseSpecialGroup(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    @patch.multiple(BaseSpecialGroup, __abstractmethods__=set())
    def test__given_vector__when_basespecialgroup_hat__then_raises_not_implemented(self):

        # Given
        instance = BaseSpecialGroup()
        vector = np.array([1, 2, 3], dtype=np.float32)

        # When and then
        with self.assertRaises(NotImplementedError):
            instance.hat(vector)

    @patch.multiple(BaseSpecialGroup, __abstractmethods__=set())
    def test__given_vector__when_basespecialgroup_exp__then_raises_not_implemented(self):

        # Given
        instance = BaseSpecialGroup()
        vector = np.array([1, 2, 3], dtype=np.float32)

        # When and then
        with self.assertRaises(NotImplementedError):
            instance.exp(vector)

    @patch.multiple(BaseSpecialGroup, __abstractmethods__=set())
    def test__given_vector__when_basespecialgroup_log__then_raises_not_implemented(self):

        # Given
        instance = BaseSpecialGroup()
        matrix = np.array([1, 2, 3], dtype=np.float32)

        # When and then
        with self.assertRaises(NotImplementedError):
            instance.log(matrix)

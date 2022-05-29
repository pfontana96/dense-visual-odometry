from unittest import TestCase
import pytest

import logging
import numpy as np

from dense_visual_odometry.utils.numpy_cache import np_cache


logger = logging.getLogger(__name__)
DOING_COMPUTATION_MESSAGE = "Doing computation.."


@np_cache
def dummy_numpy_function(array: np.ndarray):
    logger.info(DOING_COMPUTATION_MESSAGE)
    return array * 2


class TestNumpyCache(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def test__given_same_input_twice__when_np_cache__then_called_cached(self):

        # Given
        array = np.arange(5)

        # When
        with self._caplog.at_level(logging.INFO):
            result = dummy_numpy_function(array)
            result_1 = dummy_numpy_function(array)

        # Then
        messages = [record.message for record in self._caplog.records]
        self.assertListEqual(messages, [DOING_COMPUTATION_MESSAGE])
        np.testing.assert_array_equal(result, result_1)
        dummy_numpy_function.cache_clear()

    def test_given_different_inputs__when_np_cache__then_called_twice(self):

        # Given
        array = np.arange(5)
        array_1 = np.array([5, 4, 3, 2, 1])

        # When
        with self._caplog.at_level(logging.INFO):
            dummy_numpy_function(array)
            dummy_numpy_function(array_1)

        # Then
        messages = [record.message for record in self._caplog.records]
        self.assertListEqual(messages, [DOING_COMPUTATION_MESSAGE, DOING_COMPUTATION_MESSAGE])
        dummy_numpy_function.cache_clear()

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
        array = np.arange(5).reshape(-1, 1)

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
        array = np.arange(5).reshape(-1, 1)
        array_1 = np.array([5, 4, 3, 2, 1]).reshape(-1, 1)

        # When
        with self._caplog.at_level(logging.INFO):
            dummy_numpy_function(array)
            dummy_numpy_function(array_1)

        # Then
        messages = [record.message for record in self._caplog.records]
        self.assertListEqual(messages, [DOING_COMPUTATION_MESSAGE, DOING_COMPUTATION_MESSAGE])
        dummy_numpy_function.cache_clear()

    def test__given_same_some_numpy_args_and_kwargs__when_np_cache__then_called_once(self):

        # Given
        @np_cache
        def dummy_numpy_method(arr: np.ndarray, notarr: int, key: bool = False):
            logger.info(DOING_COMPUTATION_MESSAGE)
            return arr + notarr * int(key)

        array = np.array([1, 2, 3, 4]).reshape(-1, 1)
        notarr = 5
        key = True

        # When
        with self._caplog.at_level(logging.INFO):
            result = dummy_numpy_method(array, notarr, key=key)
            result_1 = dummy_numpy_method(array, notarr, key=key)

        # Then
        messages = [record.message for record in self._caplog.records]
        self.assertListEqual(messages, [DOING_COMPUTATION_MESSAGE])
        np.testing.assert_array_equal(result, result_1)
        dummy_numpy_method.cache_clear()

    def test__given_same_some_args_and_numpy_kwargs__when_np_cache__then_called_once(self):

        # Given
        @np_cache
        def dummy_numpy_method(notarr: int, arr: np.ndarray, key: bool = False):
            logger.info(DOING_COMPUTATION_MESSAGE)
            return arr + notarr * int(key)

        array = np.array([1, 2, 3, 4]).reshape(-1, 1)
        notarr = 5
        key = True

        # When
        with self._caplog.at_level(logging.INFO):
            result = dummy_numpy_method(notarr, arr=array, key=key)
            result_1 = dummy_numpy_method(notarr, arr=array, key=key)

        # Then
        messages = [record.message for record in self._caplog.records]
        self.assertListEqual(messages, [DOING_COMPUTATION_MESSAGE])
        np.testing.assert_array_equal(result, result_1)
        dummy_numpy_method.cache_clear()

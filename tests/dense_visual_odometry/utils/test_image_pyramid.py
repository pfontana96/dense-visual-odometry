import numpy as np
import cv2
import logging
import pytest

from dense_visual_odometry.utils.image_pyramid import ImagePyramid, ImagePyramidError

from unittest import TestCase
from unittest.mock import patch


class TestImagePyramid(TestCase):

    def setUp(self) -> None:
        self.maxDiff = None

        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))
        self.image = rs.randint(0, 255, size=(160, 160, 3)).astype(np.uint8)  # Random RGB image

    def tearDown(self) -> None:
        return super().tearDown()

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    @patch("cv2.pyrDown")
    def test__given_a_image_and_levels__when_init__then_called_right_amount(self, pyrdown_mock):

        # Given
        nb_levels = 5

        # When
        with self._caplog.at_level(logging.ERROR):
            pyramid = ImagePyramid(self.image, nb_levels)

        # Then
        self.assertEqual(pyrdown_mock.call_count, nb_levels - 1)
        self.assertEqual(len(pyramid.pyramid), nb_levels)
        for level in range(nb_levels):
            self.assertIsNotNone(pyramid.get(level))
        self.assertEqual(self._caplog.records, [])  # No errors logged

    def test__given_a_image_and_levels__when_init__then_right_resolution(self):

        # Given
        nb_levels = 3

        # When
        with self._caplog.at_level(logging.ERROR):
            pyramid = ImagePyramid(self.image, nb_levels)

        # Then
        self.assertEqual(len(pyramid.pyramid), nb_levels)
        self.assertEqual(pyramid.get(0).shape, (160, 160, 3))
        self.assertEqual(pyramid.get(1).shape, (80, 80, 3))
        self.assertEqual(pyramid.get(2).shape, (40, 40, 3))
        self.assertEqual(self._caplog.records, [])  # No errors logged

    def test__given_wrong_type_image__when_init__then_raises_imagepyramid(self):

        # Given
        nb_levels = 5
        image = "This is not an image"

        # When and then
        with self._caplog.at_level(logging.ERROR):
            with self.assertRaises(ImagePyramidError):
                _ = ImagePyramid(image, nb_levels)

        # Assert that the OpenCV error was logged
        self.assertNotEqual(self._caplog.records, [])

    def test__non_valid_level__when_init__then_raises_type(self):

        # Given
        nb_levels = -1

        # When and then
        with self._caplog.at_level(logging.ERROR):
            with self.assertRaises(AssertionError):
                _ = ImagePyramid(self.image, nb_levels)

    def test__given_a_non_valid_level__when_get__then_raises_assertion(self):

        # Given
        level = -1
        pyramid = ImagePyramid(self.image, 3)

        # When and Then
        with self._caplog.at_level(logging.ERROR):
            with self.assertRaises(AssertionError):
                _ = pyramid.get(level)

    def test__given_a_valid_level__when_get__then_ok(self):

        # Given
        level = 2
        pyramid = ImagePyramid(self.image, 4)

        # When
        with self._caplog.at_level(logging.ERROR):
            result = pyramid.get(level)

        # Then
        np.testing.assert_equal(result, cv2.pyrDown(cv2.pyrDown(self.image)))
        self.assertEqual(self._caplog.records, [])  # No errors logged

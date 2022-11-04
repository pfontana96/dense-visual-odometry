import numpy as np
import logging
import pytest

from dense_visual_odometry.utils.image_pyramid import ImagePyramid, ImagePyramidError, CoarseToFineMultiImagePyramid

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

    @patch("dense_visual_odometry.utils.image_pyramid.pyrDownMedianSmooth")
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
            self.assertIsNotNone(pyramid[level])
        self.assertEqual(self._caplog.records, [])  # No errors logged

    def test__given_a_image_and_levels__when_init__then_right_resolution(self):

        # Given
        nb_levels = 3

        # When
        with self._caplog.at_level(logging.ERROR):
            pyramid = ImagePyramid(self.image, nb_levels)

        # Then
        self.assertEqual(len(pyramid.pyramid), nb_levels)
        self.assertEqual(pyramid[0].shape, (160, 160, 3))
        self.assertEqual(pyramid[1].shape, (80, 80, 3))
        self.assertEqual(pyramid[2].shape, (40, 40, 3))
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
            with self.assertRaises(IndexError):
                _ = pyramid[level]


class TestCoarseToFineMultiImagePyramid(TestCase):

    def setUp(self) -> None:
        self.maxDiff = None

        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))
        self.nb_of_images = 3
        self.image_shape = (160, 160, 3)
        self.images = [rs.randint(0, 255, size=self.image_shape).astype(np.uint8) for _ in range(self.nb_of_images)]

    def test__given_coarsetofinepyramid__when_iter__then_ok(self):
        # Given
        levels = 3
        pyramid = CoarseToFineMultiImagePyramid(images=self.images, levels=levels)

        # When + Then
        import logging
        logger = logging.getLogger(__name__)
        i = levels - 1
        for image1, image2, image3 in pyramid:
            logger.info("{} level".format(i))
            self.assertIsInstance(image1, np.ndarray)
            self.assertEqual(image1.shape[:2], tuple(np.array(self.image_shape[:2]) / (2 ** i)))

            self.assertIsInstance(image2, np.ndarray)
            self.assertEqual(image2.shape[:2], tuple(np.array(self.image_shape[:2]) / (2 ** i)))

            self.assertIsInstance(image3, np.ndarray)
            self.assertEqual(image3.shape[:2], tuple(np.array(self.image_shape[:2]) / (2 ** i)))

            i -= 1

        self.assertEqual(i, -1)

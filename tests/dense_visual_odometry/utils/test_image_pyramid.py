import numpy as np

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

    @patch("dense_visual_odometry.utils.image_pyramid.pyrDownMedianSmooth")
    def test__given_a_image_and_levels__when_init__then_called_right_amount(self, pyrdown_mock):

        # Given
        nb_levels = 5

        # When
        pyramid = ImagePyramid(image=self.image, levels=nb_levels)

        # Then
        self.assertEqual(pyrdown_mock.call_count, nb_levels - 1)
        self.assertEqual(len(pyramid._pyramid), nb_levels)
        for level in range(nb_levels):
            self.assertIsNotNone(pyramid.at(level))

    def test__given_a_image_and_levels__when_init__then_right_resolution(self):

        # Given
        nb_levels = 3

        # When
        pyramid = ImagePyramid(image=self.image, levels=nb_levels)

        # Then
        self.assertEqual(len(pyramid._pyramid), nb_levels)
        self.assertEqual(pyramid.at(0).shape, (160, 160, 3))
        self.assertEqual(pyramid.at(1).shape, (80, 80, 3))
        self.assertEqual(pyramid.at(2).shape, (40, 40, 3))

    def test__given_wrong_type_image__when_init__then_raises_imagepyramid(self):

        # Given
        nb_levels = 5
        image = "This is not an image"

        # When and then
        with self.assertRaises(ImagePyramidError):
            _ = ImagePyramid(image=image, levels=nb_levels)

    def test__non_valid_level__when_init__then_raises_type(self):

        # Given
        nb_levels = -1

        # When and then
        with self.assertRaises(ValueError):
            _ = ImagePyramid(image=self.image, levels=nb_levels)

    def test__given_a_non_valid_level__when_get__then_raises_assertion(self):

        # Given
        level = -1
        pyramid = ImagePyramid(image=self.image, levels=3)

        # When and Then
        with self.assertRaises(IndexError):
            _ = pyramid.at(level)

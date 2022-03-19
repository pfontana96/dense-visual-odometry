from unittest import TestCase

from dense_visual_odometry.core import dummy_function


class TestCore(TestCase):

    def setUp(self) -> None:
        self.maxDiff = None

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_a_and_b__when_dummy_function__then_ok(self):
        # Given
        a = 2
        b = 4

        # When
        c = dummy_function(a, b)

        # Then
        self.assertEqual(6, c)

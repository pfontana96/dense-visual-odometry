from unittest import TestCase
import pytest

import logging

from dense_visual_odometry.log import set_root_logger


logger = logging.getLogger(__name__)


class TestSetRootLogger(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def test__given_no_verbose__when_set_root_logger__then_no_debug(self):

        # Given
        set_root_logger()

        # When + Then
        self.assertFalse(logger.isEnabledFor(logging.DEBUG))
        self.assertTrue(logger.isEnabledFor(logging.INFO))

    def test__given_verbose__when_set_root_logger__then_debug(self):

        # Given
        set_root_logger(verbose=True)

        # When + Then
        self.assertTrue(logger.isEnabledFor(logging.DEBUG))

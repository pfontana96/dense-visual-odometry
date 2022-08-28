from pathlib import Path

import cv2

import pytest


_TEST_DATADIR = Path(__file__).resolve().parent / "test_data"


@pytest.fixture
def load_rgb_image(image_name: str):
    filename = _TEST_DATADIR / "rgb" / image_name
    return cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)


@pytest.fixture
def load_depth_image(image_name: str):
    filename = _TEST_DATADIR / "depth" / image_name
    return cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH)


@pytest.fixture
def load_camera_intrinsics_file():
    filepath = _TEST_DATADIR / "test_camera_intrinsics.yaml"
    return filepath

from pathlib import Path
import json

import numpy as np
import cv2

import pytest


_TEST_DATADIR = Path(__file__).resolve().parent / "test_data"


def _load_grayscale_image(filepath: str):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def _load_bgr_image(filepath: str):
    return cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)


def _load_depth_image(filepath: str):
    return cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)


@pytest.fixture
def load_camera_intrinsics_file():
    filepath = _TEST_DATADIR / "test_camera_intrinsics.yaml"
    return filepath


@pytest.fixture
def load_test_benchmark():

    # Load ground truth
    with (_TEST_DATADIR / "ground_truth.json").open("r") as fp:
        ground_truth_data = json.load(fp)

    rgb_images = []
    depth_images = []
    ground_truth_transformations = []
    for value in ground_truth_data.values():
        rgb_images.append(_load_grayscale_image(str(_TEST_DATADIR / value["rgb"])))
        depth_images.append(_load_depth_image(str(_TEST_DATADIR / value["depth"])))
        ground_truth_transformations.append(np.array(value["transformation"]))

    return rgb_images, depth_images, ground_truth_transformations


@pytest.fixture
def load_single_benchmark_case(request):

    assert isinstance(request.param, int), "Expected request parameter to be 'int', got '{}' instead".format(
        type(request.param)
    )
    index = request.param

    # Load ground truth
    with (_TEST_DATADIR / "ground_truth.json").open("r") as fp:
        ground_truth_data = json.load(fp)

    gray_images = []
    depth_images = []
    transformations = []
    for i in range(index, index + 2):
        gray_images.append(_load_grayscale_image(str(_TEST_DATADIR / ground_truth_data[str(i)]["rgb"])))
        depth_images.append(_load_depth_image(str(_TEST_DATADIR / ground_truth_data[str(i)]["depth"])))
        transformations.append(np.array(ground_truth_data[str(i)]["transformation"]))

    return gray_images, depth_images, transformations

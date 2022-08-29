from pathlib import Path
import json

import numpy as np
import cv2

import pytest


_TEST_DATADIR = Path(__file__).resolve().parent / "test_data"


@pytest.fixture
def load_rgb_image(filepath: str):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


@pytest.fixture
def load_depth_image(filepath: str):
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
        rgb_images.append(load_rgb_image(str(_TEST_DATADIR / value["rgb"])))
        depth_images.append(load_rgb_image(str(_TEST_DATADIR / value["depth"])))
        ground_truth_transformations.append(np.array(value["transformation"]))

    return rgb_images, depth_images, ground_truth_transformations

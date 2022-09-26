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
def load_single_benchmark_case():

    # Load ground truth
    with (_TEST_DATADIR / "ground_truth.json").open("r") as fp:
        ground_truth_data = json.load(fp)

    index = "5"  # Randomly chosen
    ground_truth = ground_truth_data[index]

    gray_image = _load_grayscale_image(str(_TEST_DATADIR / ground_truth["rgb"]))
    depth_image = _load_depth_image(str(_TEST_DATADIR / ground_truth["depth"]))
    transformation = np.array(ground_truth["transformation"])

    return gray_image, depth_image, transformation

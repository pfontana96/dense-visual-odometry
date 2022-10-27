from pathlib import Path
import json
import logging
from time import time

import numpy as np
import cv2

from dense_visual_odometry.core import KerlDVO
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.log import set_root_logger
from dense_visual_odometry.utils.lie_algebra import SE3


logger = logging.getLogger(__name__)


def load_benchmark(data_path: Path = None):
    if data_path is None:
        # Use test benchmark
        data_path = Path(__file__).resolve().parent.parent / "tests" / "test_data"
    with (data_path / "ground_truth.json").open("r") as fp:
        data = json.load(fp)

    camera_model = RGBDCameraModel.load_from_yaml(data_path / "camera_intrinsics.yaml")

    transformations = []
    rgb_images = []
    depth_images = []
    additional_info = {"rgb": [], "depth": [], "camera_intrinsics": str(data_path / "camera_intrinsics.yaml")}
    for value in data.values():
        transformations.append(SE3.log(np.array(value["transformation"])))
        rgb_images.append(
            cv2.cvtColor(cv2.imread(str(data_path / value["rgb"]), cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)
        )
        depth_images.append(cv2.imread(str(data_path / value["depth"]), cv2.IMREAD_ANYDEPTH))

        additional_info["rgb"].append(str(data_path / value["rgb"]))
        additional_info["depth"].append(str(data_path / value["depth"]))

    return transformations, rgb_images, depth_images, camera_model, additional_info


def main():
    set_root_logger(verbose=False)

    gt_transforms, rgb_images, depth_images, camera_model, additional_info = load_benchmark()

    dvo = KerlDVO(camera_model=camera_model, initial_pose=gt_transforms[0], levels=5)

    steps = []
    errors = []
    for i, (rgb_image, depth_image, gt_transform) in enumerate(zip(rgb_images, depth_images, gt_transforms)):

        s = time()
        dvo.step(rgb_image, depth_image)
        e = time()

        # Error is only the euclidean distance (not taking rotation into account)
        error = np.linalg.norm(dvo.current_pose[:3] - gt_transform[:3])
        logger.info("[Frame {} ({:.3f} s)] Error: {} m".format(i + 1, e - s, error))

        steps.append(dvo.current_pose.reshape(-1).tolist())
        errors.append(float(error))

    # Dump results
    report = {"estimated_transformations": steps, "errors": errors}
    report.update(additional_info)

    output_file = Path(__file__).resolve().parent.parent / "data" / "report.json"

    with output_file.open("w") as fp:
        json.dump(report, fp, indent=3)


if __name__ == "__main__":
    main()

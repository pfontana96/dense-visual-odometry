from pathlib import Path
import json
import logging

import numpy as np
import cv2

from dense_visual_odometry.core import DenseVisualOdometry
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.log import set_root_logger


logger = logging.getLogger(__name__)


USE_TEST_DATA = True


def main():
    set_root_logger(verbose=True)

    if USE_TEST_DATA:
        camera_model_config = Path(__file__).resolve().parent.parent / "tests" / "test_data" / "test_camera_intrinsics.yaml"
        depth_images_path = Path(__file__).resolve().parent.parent / "tests" / "test_data" / "depth"
        rgb_images_path = Path(__file__).resolve().parent.parent / "tests" / "test_data" / "rgb"
    
    else:
        camera_model_config = Path(__file__).resolve().parent.parent / "data" / "camera_intrinsics.yaml"
        depth_images_path = Path(__file__).resolve().parent.parent / "data" / "depth"
        rgb_images_path = Path(__file__).resolve().parent.parent / "data" / "rgb"

    camera_model = RGBDCameraModel.load_from_yaml(camera_model_config)

    init_pose = np.array([[0.0], [0.0], [0.16], [-1.33658619],  [1.33658135], [-1.12156232]], dtype=np.float32)

    dvo = DenseVisualOdometry(camera_model=camera_model, initial_pose=init_pose, levels=5)

    n = 10  # Hardcoded because of available data

    steps = []
    for i in range(n):
        color = cv2.imread(str(rgb_images_path / "{}.png".format(i+1)))
        depth = cv2.imread(str(depth_images_path / "{}.png".format(i+1)), cv2.IMREAD_ANYDEPTH)
        # depth_cmap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.09), cv2.COLORMAP_JET)
        dvo.step(color_image=color, depth_image=depth)
        steps.append(dvo.current_pose.tolist())

    # Dump results
    report = {"estimated_transformations": steps}
    output_file = Path(__file__).resolve().parent.parent / "data" / "report.json"

    json.dump(report, output_file.open("w"), indent=4)


if __name__ == "__main__":
    main()

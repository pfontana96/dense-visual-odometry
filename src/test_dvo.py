from pathlib import Path
import json
import logging
from time import time
from argparse import ArgumentParser
from typing import Union

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from tqdm import tqdm

from dense_visual_odometry.core import get_dvo
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.log import set_root_logger
from dense_visual_odometry.utils.lie_algebra import Se3, So3
from dense_visual_odometry.utils.image_pyramid import pyrDownMedianSmooth


logger = logging.getLogger(__name__)


_SUPPORTED_BENCHMARKS = ["tum-fr1", "test"]


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("benchmark", type=str, choices=_SUPPORTED_BENCHMARKS, help="Benchmark to run")
    parser.add_argument("-d", "--data-dir", type=str, help="Path to data", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-s", "--size", type=int, default=None,
        help="Number of data samples to use (first 'size' samples are selected)"
    )
    parser.add_argument("-m", "--method", type=str, default="kerl", help="Dense visual odometry method to use")
    parser.add_argument("-c", "--config-file", type=str, help="Dense visual odometry method's configuraion JSON")
    parser.add_argument("-p", "--pyr-down", action="store_true", help="If to downsample dataset image size")

    args = parser.parse_args()

    set_root_logger(verbose=args.verbose)

    gt_transforms, rgb_images, depth_images, camera_model, additional_info = load_benchmark(
        type=args.benchmark, data_dir=args.data_dir, size=args.size
    )

    init_pose = gt_transforms[0] if gt_transforms[0] is not None else Se3.identity()

    config = {}
    if args.config_file is not None:
        config = _load_config(args.config_file)

    dvo = get_dvo(args.method, camera_model=camera_model, init_pose=init_pose, **config)

    return dvo, gt_transforms, rgb_images, depth_images, additional_info


def load_benchmark(type: str, data_dir: str = None, size: int = None, pyr_down: bool = False):
    if type == "tum-fr1":
        if data_dir is None:
            raise ValueError("When running 'tum-fr1' path to data (-d) should be specified")

        data_dir = Path(data_dir).resolve()

        if not data_dir.is_dir():
            raise FileNotFoundError("Could not find data dir at '{}'".format(str(data_dir)))

        camera_intrinsics_file = Path(__file__).resolve().parent.parent / "tests/test_data/camera_intrinsics.yaml"

        gt_transforms, rgb_images, depth_images, camera_model, additional_info = _load_tum_benchmark(
            data_path=data_dir, camera_intrinsics_file=camera_intrinsics_file, pyr_down=pyr_down, size=size
        )

    elif type == "test":
        if data_dir is not None:
            data_dir = Path(data_dir).resolve()

        gt_transforms, rgb_images, depth_images, camera_model, additional_info = _load__test_benchmark(
            data_path=data_dir, size=size
        )

    return gt_transforms, rgb_images, depth_images, camera_model, additional_info


def _load_tum_benchmark(data_path: Path, camera_intrinsics_file: Path, pyr_down: bool = False, size: int = None):
    """Loads TUM RGB-D benchmarks. See https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats

    Parameters
    ----------
    data_path : Path
        Path to dir where benchmark is downloaded.
    camera_intrinsics_file : Path
        Path to the camera intrinsics parameters.

    Returns
    -------
    List[np.ndarray] :
        List of ground truth camera poses.
    List[np.ndarray] :
        List of RGB images.
    List[np.ndarray] :
        List of depth images.
    RGBDCameraModel :
        Camera model to be used for processing dataset.
    dict :
        Dictionary containing info about where the loaded images are stored on the filesystem.
    """

    filenames = ["rgb.txt", "depth.txt", "groundtruth.txt"]
    filedata = {}

    # Load data from txt files
    for filename in tqdm(filenames, ascii=True, desc="Reading txt files"):
        filepath = data_path / filename

        if not filepath.exists():
            raise FileNotFoundError("Expected TUM RGB-D dataset to contain a file named '{}' at '{}'".format(
                filename, str(data_path)
            ))

        with filepath.open("r") as fp:
            content = fp.readlines()

        timestamps = []
        data = []
        for line in content:
            # Avoid comments
            if line.startswith('#'):
                continue

            line_stripped = line.rstrip("\r\n")
            line_stripped = line_stripped.split(" ")
            timestamps.append(float(line_stripped[0]))

            # If groundtruth then save se(3) pose
            if filename == "groundtruth.txt":
                pose = Se3(
                    So3(np.roll(np.asarray(line_stripped[4:], dtype=np.float32), 1).reshape(4, 1)),
                    np.asarray(line_stripped[1:4], dtype=np.float32).reshape(3, 1)
                )
                data.append(pose)

            else:
                image_path = data_path / line_stripped[1]
                if not image_path.exists():
                    raise FileNotFoundError("Could not find {} image at '{}'".format(filepath.stem, str(image_path)))
                data.append(str(image_path))

        filedata[filepath.stem] = {"timestamp": np.array(timestamps), "data": np.array(data)}

    # Find timestamps correspondances between depth, rgb
    logger.info("Finding closest timestamps..")
    distance = np.abs(filedata["rgb"]["timestamp"].reshape(-1, 1) - filedata["depth"]["timestamp"])
    potential_closest = distance.argmin(axis=1)

    # Avoid duplicates
    closest_found, ids = np.unique(potential_closest, return_index=True)
    rgb_timestamps = filedata["rgb"]["timestamp"][ids]
    depth_timestamps = filedata["depth"]["timestamp"][closest_found]

    rgb_images_paths = filedata["rgb"]["data"][ids]
    depth_images_paths = filedata["depth"]["data"][closest_found]

    # Find closest groundtruth (ugly average between the timestamps of rgb and dept)
    frames_timestamps = (rgb_timestamps + depth_timestamps) / 2
    closests_groundtruth = np.argmin(
        cdist(frames_timestamps.reshape(-1, 1), filedata["groundtruth"]["timestamp"].reshape(-1, 1)), axis=1
    )
    logger.info("DONE")

    gt_transforms = list(filedata["groundtruth"]["data"][closests_groundtruth])

    if (size is not None) and (size < len(rgb_images_paths)):
        gt_transforms = gt_transforms[:size]
        rgb_images_paths = rgb_images_paths[:size]
        depth_images_paths = depth_images_paths[:size]

    rgb_images = []
    depth_images = []
    for rgb_path, depth_path in tqdm(zip(rgb_images_paths, depth_images_paths), ascii=True, desc="Loading images"):
        rgb_image = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        if pyr_down:
            rgb_image = pyrDownMedianSmooth(image=rgb_image)
            depth_image = pyrDownMedianSmooth(image=depth_image)

        rgb_images.append(rgb_image)
        depth_images.append(depth_image)

    camera_model = RGBDCameraModel.load_from_yaml(camera_intrinsics_file)

    # Additional data needed for report
    additional_info = {
        "rgb": rgb_images_paths.tolist(), "depth": depth_images_paths.tolist(),
        "camera_intrinsics": str(camera_intrinsics_file)
    }

    return gt_transforms, rgb_images, depth_images, camera_model, additional_info


def _load__test_benchmark(data_path: Path = None, size: int = None):
    """Loads test benchmark (custom dataset format)

    Parameters
    ----------
    data_path : Path, optional
        Path to where the data is located, by default None. If None, then the testing dataset is loaded

    Returns
    -------
    List[np.ndarray] :
        List of ground truth camera poses.
    List[np.ndarray] :
        List of RGB images.
    List[np.ndarray] :
        List of depth images.
    RGBDCameraModel :
        Camera model to be used for processing dataset.
    dict :
        Dictionary containing info about where the loaded images are stored on the filesystem.
    """
    if data_path is None:
        # Use test benchmark
        data_path = Path(__file__).resolve().parent.parent / "tests" / "test_data"

    with (data_path / "ground_truth.json").open("r") as fp:
        data = json.load(fp)

    camera_model = RGBDCameraModel.load_from_yaml(data_path / "camera_intrinsics.yaml")

    available_gt = True
    transformations = []
    rgb_images = []
    depth_images = []
    additional_info = {"rgb": [], "depth": [], "camera_intrinsics": str(data_path / "camera_intrinsics.yaml")}
    for i, value in enumerate(data.values()):
        if available_gt:
            try:
                xi = np.array(value["transformation"]).reshape(6, 1)
                transformations.append(Se3(So3(xi[3:]), xi[:3]))
            except KeyError as e:
                logger.info("Could not find ground truth transform under '{}' at '{}'".format(
                    e, str(data_path)
                ))
                available_gt = False

        rgb_images.append(
            cv2.cvtColor(cv2.imread(str(data_path / value["rgb"]), cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)
        )
        depth_images.append(cv2.imread(str(data_path / value["depth"]), cv2.IMREAD_ANYDEPTH))

        additional_info["rgb"].append(str(data_path / value["rgb"]))
        additional_info["depth"].append(str(data_path / value["depth"]))

        if size is not None:
            if i >= (size - 1):
                logger.info("Using first '{}' data samples".format(size))
                break

    if not available_gt:
        transformations = [None] * len(rgb_images)

    return transformations, rgb_images, depth_images, camera_model, additional_info


def _load_config(path: Union[str, Path]):
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError("Could not find configuration file at '{}'".format(str(path)))

    if not path.suffix == ".json":
        raise ValueError("Expected config file to be '.json' got '{}' instead".format(path.suffix))

    with path.open("r") as fp:
        config = json.load(fp)

    return config


def main():

    dvo, gt_transforms, rgb_images, depth_images, additional_info = parse_arguments()

    steps = []
    errors = []
    for i, (rgb_image, depth_image, gt_transform) in enumerate(zip(rgb_images, depth_images, gt_transforms)):

        s = time()
        dvo.step(rgb_image, depth_image)
        e = time()

        # Error is only the euclidean distance (not taking rotation into account)
        if gt_transform is not None:
            t_error = float(np.linalg.norm(dvo.current_pose.tvec - gt_transform.tvec))
            r_error = float(np.linalg.norm(dvo.current_pose.so3.log() - gt_transform.so3.log()))
            logger.info("[Frame {} ({:.3f} s)] | Translational error: {:.4f} m | Rotational error: {:.4f}".format(
                i + 1, e - s, t_error, r_error
            ))
        else:
            t_error = "N/A"
            r_error = "N/A"
            logger.info("[Frame {} ({:.3f} s)]".format(i + 1, e - s))

        steps.append(dvo.current_pose.log().flatten().tolist())
        errors.append(t_error)

    # Dump results
    report = {"estimated_transformations": steps, "errors": errors}
    report.update(additional_info)

    output_file = Path(__file__).resolve().parent.parent / "data" / "report.json"

    with output_file.open("w") as fp:
        json.dump(report, fp, indent=3)


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from pathlib import Path
import json
import logging
from typing import List, Type
import time
import copy

import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

from dense_visual_odometry.utils.lie_algebra.special_euclidean_group import Se3, So3
from dense_visual_odometry.log import set_root_logger
from dense_visual_odometry.camera_model import RGBDCameraModel

from test_dvo import load_benchmark, _SUPPORTED_BENCHMARKS


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="Command", dest="cmd")

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-p", "--plot-trajectory", action="store_true", help="Flag to display trajectory plots in x, y and z"
    )
    parent_parser.add_argument(
        "-a", "--absolute", action="store_true",
        help="Bool flag indicating if transforms are absolute or relative to previous one"
    )
    parent_parser.add_argument(
        "-s", "--size", type=int, default=None,
        help="Number of data samples to use (first 'size' samples are selected)"
    )
    parent_parser.add_argument("-f", "--fps", type=int, default=5, help="fps to use in animation")

    report_parser = subparsers.add_parser(
        "report", parents=[parent_parser], help="Visualize from a JSON report generated"
    )
    report_parser.add_argument("report_file", type=str, help="Path to report file (.JSON)")

    benchmark_parser = subparsers.add_parser(
        "benchmark", parents=[parent_parser], help="Visualize from a benchmark dir"
    )
    benchmark_parser.add_argument("type", type=str, choices=_SUPPORTED_BENCHMARKS, help="Type of benchmark to load")
    benchmark_parser.add_argument("-d", "--data-path", type=str, help="Path to benchmark dir", default=None)

    args = parser.parse_args()

    if args.cmd == "report":
        report = Path(args.report_file).resolve()
        if not report.exists():
            logger.error("Could not find report file at '{}'".format(str(report)))
            raise FileNotFoundError("Could not find report file at '{}'".format(str(report)))

        if not report.suffix == ".json":
            raise ValueError("Expected 'report' extension to be '.json' got '{}' instead".format(report.suffix))

        transforms, rgb_images, depth_images, camera_model = load_from_report(report)

    elif args.cmd == "benchmark":
        data = args.data_path

        if data is not None:
            data = Path(args.data_path).resolve()
            if not data.is_dir():
                raise ValueError("Could not find benchmark dir at '{}'".format(str(data)))

        transforms, rgb_images, depth_images, camera_model, _ = load_benchmark(args.type, data, args.size)

    return transforms, rgb_images, depth_images, camera_model, args.plot_trajectory, args.absolute, args.fps


def load_from_report(report_path: Path):
    if not report_path.exists():
        logger.error("Could not find report file at '{}'".format(str(report_path)))
        raise FileNotFoundError("Could not find report file at '{}'".format(str(report_path)))

    if not report_path.suffix == ".json":
        raise ValueError("Expected 'report' extension to be '.json' got '{}' instead".format(report_path.suffix))

    with report_path.open("r") as fp:
        data = json.load(fp)

    estimated_transforms = []
    rgb_images = []
    depth_images = []
    try:
        for rgb_image_path, depth_image_path, transform in tqdm(zip(
            data["rgb"], data["depth"], data["estimated_transformations"]
        ), ascii=True, desc="Loading images from report.."):
            rgb_images.append(cv2.cvtColor(cv2.imread(rgb_image_path, cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB))
            depth_images.append(cv2.imread(str(depth_image_path), cv2.IMREAD_ANYDEPTH))
            xi = np.array(transform, dtype=np.float32).reshape(6, 1)
            estimated_transforms.append(Se3(So3(xi[3:]), xi[3:]))

        camera_model = RGBDCameraModel.load_from_yaml(Path(data["camera_intrinsics"]).resolve())

    except KeyError as e:
        raise ValueError("Expected report to contain '{}' field".format(e))

    return estimated_transforms, rgb_images, depth_images, camera_model


def main():

    set_root_logger(verbose=False)

    transformations, rgb_images, depth_images, camera_model, plot_t, absolute, fps = parse_arguments()

    animate3d(rgb_images, depth_images, transformations, camera_model, absolute_transforms=absolute, fps=fps)


def animate3d(
    rgb_images: List[np.ndarray], depth_images: List[np.ndarray], transforms: List[np.ndarray],
    camera_model: Type[RGBDCameraModel], absolute_transforms: bool = False, fps: int = 1
):
    period = 1 / fps

    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # Black background

    pcd = o3d.geometry.PointCloud()
    camera_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    max_distance = 5  # [m]

    for i, (rgb_image, depth_image, xi_t) in enumerate(zip(rgb_images, depth_images, transforms)):
        start = time.time()

        # Filter noisy points of sensor
        depth_image[(depth_image * camera_model.depth_scale) > max_distance] = 0

        # Update camera pose
        if absolute_transforms and (i != 0):
            xi_t_1 = transforms[i - 1]
            xi = xi_t.inverse() * xi_t_1
        else:
            # xi = transform.copy()
            xi = xi_t

        camera_pose.transform(xi.exp())
        # camera_pose.rotate(xi.so3.exp())
        # camera_pose.translate(xi.tvec)
        pointcloud_xyz, mask = camera_model.deproject(depth_image=depth_image, return_mask=True)

        # Transform pointcloud from camera reference frame to world
        pointcloud_xyz = np.dot(xi_t.exp(), pointcloud_xyz)

        pointcloud_colors = rgb_image[mask]

        # Update Pointcloud
        pcd.points = o3d.utility.Vector3dVector(pointcloud_xyz[:3, :].T)
        pcd.colors = o3d.utility.Vector3dVector(pointcloud_colors / 255.0)

        if i == 0:
            # Change channels on initial_pcd to being able to differenciate it from other pointclouds as to use it as
            # a visual reference
            initial_pcd = copy.deepcopy(pcd)
            initial_pcd.colors = o3d.utility.Vector3dVector(np.roll(pointcloud_colors, 1, axis=1) / 255.0)

            vis.add_geometry(initial_pcd)
            vis.add_geometry(pcd)
            vis.add_geometry(camera_pose)
        else:
            vis.update_geometry(pcd)
            vis.update_geometry(camera_pose)

        vis.poll_events()
        vis.update_renderer()

        end = time.time()
        sleep_time = period - (end - start)
        if sleep_time > 0.0:
            time.sleep(sleep_time)

    vis.destroy_window()
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)


if __name__ == "__main__":
    main()

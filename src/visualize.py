from argparse import ArgumentParser
from pathlib import Path
import json
import logging

import numpy as np

from dense_visual_odometry.utils.lie_algebra.special_euclidean_group import SE3
from dense_visual_odometry.log import set_root_logger


logger = logging.getLogger(__name__)


def create_fixed_transformations():
    x_final = 5
    tras = np.arange(0, x_final, 0.1)
    zeros = np.zeros_like(tras)
    xi_t = np.vstack((tras, zeros, zeros, zeros, zeros, zeros))

    return xi_t


def load_transformation_from_test_benchmark():
    with (Path(__file__).resolve().parent.parent / "tests" / "test_data" / "ground_truth.json").open("r") as fp:
        data = json.load(fp)

    transformations = []
    for value in data.values():
        transformations.append(SE3.log(np.array(value["transformation"])))

    return transformations


class TF(object):
    def __init__(self, xi: np.ndarray = np.zeros((6, 1))):
        self.xi = xi

    def update(self, xi):
        self.xi = xi

    def get_matrix(self):
        return SE3.exp(self.xi)

    @staticmethod
    def quiver_data_to_segments(x, y, z, u, v, w):
        return [[[xi, yi, zi], [ui, vi, wi]] for (xi, yi, zi, ui, vi, wi) in zip(x, y, z, u, v, w)]

    def get_quiver_data(self):

        origin = np.array([0, 0, 0, 1], dtype=np.float32).reshape(4, 1)  # Homogeneous coordinate
        x_axis = np.array([1, 0, 0, 1], dtype=np.float32).reshape(4, 1)  # Homogeneous coordinate
        y_axis = np.array([0, 1, 0, 1], dtype=np.float32).reshape(4, 1)  # Homogeneous coordinate
        z_axis = np.array([0, 0, 1, 1], dtype=np.float32).reshape(4, 1)  # Homogeneous coordinate

        T = SE3.exp(self.xi)
        logger.info("T:\n{}".format(T))

        origin = np.dot(T, origin)
        x_axis = np.dot(T, x_axis)
        y_axis = np.dot(T, y_axis)
        z_axis = np.dot(T, z_axis)

        x = np.repeat(origin[0], 3)
        y = np.repeat(origin[1], 3)
        z = np.repeat(origin[2], 3)

        u = np.array([x_axis[0], y_axis[0], z_axis[0]]).flatten()
        v = np.array([x_axis[1], y_axis[1], z_axis[1]]).flatten()
        w = np.array([x_axis[2], y_axis[2], z_axis[2]]).flatten()

        return x, y, z, u, v, w


def animate_func(i, tf, Q, poses):

    logger.info("Pose({}): {}".format(i, poses[i]))
    tf.update(poses[i])
    x, y, z, u, v, w = tf.get_quiver_data()

    segments = TF.quiver_data_to_segments(x, y, z, u, v, w)

    Q.set_segments(segments)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("report", type=str, help="Path to report file (.JSON)")
    parser.add_argument("-a", "--animate", action="store_true")

    args = parser.parse_args()

    report = Path(args.report).resolve()
    if not report.exists():
        logger.error("Could not find report file at '{}'".format(str(report)))
        raise FileNotFoundError("Could not find report file at '{}'".format(str(report)))

    if not report.suffix == ".json":
        raise ValueError("Expected 'report' extension to be '.json' got '{}' instead".format(report.suffix))

    logger.info("Animating: {}".format(args.animate))

    # Look for transformations
    data = json.load(report.open("r"))

    result = []
    try:
        transformations = data["estimated_transformations"]
        for transformation in transformations:
            result.append(np.array(transformation, dtype=np.float32))

    except KeyError as e:
        raise ValueError("Expected report to contain '{}' field".format(e))

    return result, args.animate


def main():
    import matplotlib.pyplot as plt
    from matplotlib import animation

    set_root_logger(verbose=False)

    transformations, animate = parse_arguments()
    # transformations = []
    # xi_mat = create_fixed_transformations()
    # for i in range(xi_mat.shape[1]):
    #     transformations.append(xi_mat[:, i].reshape(6, 1))

    transformations = load_transformation_from_test_benchmark()

    logger.info(np.array(transformations).shape)

    if animate:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)

        # you need to set blit=False, or the first set of arrows never gets
        # cleared on subsequent frames
        tf = TF()
        x, y, z, u, v, w = tf.get_quiver_data()
        Q = ax.quiver(
            x, y, z, u, v, w,
            colors=["red", "green", "blue"],
            length=0.5, normalize=True
        )

        nb_frames = len(transformations)

        anim = animation.FuncAnimation(  # noqa: F841
            fig, animate_func, nb_frames, fargs=(tf, Q, transformations), interval=200, blit=False, repeat=False
        )

        fig.tight_layout()

    else:
        transformations = np.hstack(transformations)

        fig, axs = plt.subplots(2, 3)
        x = np.arange(transformations[0, :].size)

        axs[0, 0].set_title("X")
        axs[0, 0].plot(x, transformations[0, :])

        axs[0, 1].set_title("Y")
        axs[0, 1].plot(x, transformations[1, :])

        axs[0, 2].set_title("Z")
        axs[0, 2].plot(x, transformations[2, :])

        axs[1, 0].set_title("roll")
        axs[1, 0].plot(x, transformations[3, :] * (180 / np.pi))

        axs[1, 1].set_title("pitch")
        axs[1, 1].plot(x, transformations[4, :] * (180 / np.pi))

        axs[1, 2].set_title("yaw")
        axs[1, 2].plot(x, transformations[5, :] * (180 / np.pi))

    plt.show()


if __name__ == "__main__":
    main()

from dense_visual_odometry.core.robust_dense_visual_odometry.gpu_robust_dense_visual_odometry import RobustDVOGPU
from dense_visual_odometry.core.robust_dense_visual_odometry.cpu_robust_dense_visual_odometry import RobustDVOCPU


def robust_dvo_factory(use_gpu: bool = False, **kwargs):
    """ Returns a Kerl's implementation
    """
    if use_gpu:
        return RobustDVOGPU(**kwargs)

    else:
        # CPU implementation does not depend on a fixed resolution as it does not needs to preallocate memory
        try:
            kwargs.pop("width")

        except KeyError:
            pass

        try:
            kwargs.pop("height")

        except KeyError:
            pass

        return RobustDVOCPU(**kwargs)

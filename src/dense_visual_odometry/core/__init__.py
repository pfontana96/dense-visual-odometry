from dense_visual_odometry.core.kerl import KerlDVO  # noqa
from dense_visual_odometry.core.loftr import LoFTRDVO  # noqa
from dense_visual_odometry.camera_model import RGBDCameraModel

import numpy as np


_SUPPORTED_METHODS = {
    "kerl": KerlDVO,
    "loftr": LoFTRDVO
}


def get_dvo(method: str, camera_model: RGBDCameraModel, init_pose: np.ndarray, **kwargs):
    """Factory for getting a Dense Visual Odometry approach

    Parameters
    ----------
    method : str
        Name of method.
    camera_model : RGBDCameraModel
        Camera model to use.
    init_pose : np.ndarray
        Initial pose expressed as a SE(3) 6x1 array.
    """

    if method not in _SUPPORTED_METHODS.keys():
        raise ValueError("Not supported method '{}', available options are '{}'".format(
            method, list(_SUPPORTED_METHODS.keys())
        ))

    try:
        result = _SUPPORTED_METHODS[method](camera_model=camera_model, initial_pose=init_pose, **kwargs)
    except Exception as e:
        raise ValueError((
            "Could not dynamically load method '{}' with parameters '{}'".format(method, kwargs),
            ", got the following exception: {}".format(e)
        ))

    return result

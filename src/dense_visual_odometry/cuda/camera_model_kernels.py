from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np

@cuda.jit
def deproject_kernel(
    depth_image: DeviceNDArray, fx: float, fy: float, cx: float, cy: float, depth_scale: float,
    mask: DeviceNDArray, pointcloud: DeviceNDArray
):
    ty, tx = cuda.grid(2)
    height, width = depth_image.shape

    z = depth_image[ty, tx] * depth_scale

    mask[ty, tx] = mask[ty, tx] and (z != 0)

    tid = ty * width + tx

    if tid >= height * width:
        return

    if not mask[ty, tx]:
        pointcloud[:3, tid] = np.nan
        return

    pointcloud[0, tid] = (tx - cx) * z / fx
    pointcloud[1, tid] = (ty - cy) * z / fy
    pointcloud[2, tid] = z
    
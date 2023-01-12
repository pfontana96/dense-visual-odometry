import math

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np


@cuda.jit('float32(uint8[:,:], int32, int32, int32, int32, boolean)', device=True)
def compute_gradients(image: DeviceNDArray, x: int, y: int, height: int, width: int, x_direction: bool) -> float:
    
    if x_direction:
        prev_value = image[y, max(x - 1, 0)]
        next_value = image[y, min(x + 1, width - 1)]

    else:
        prev_value = image[max(y - 1, 0), x]
        next_value = image[min(y + 1, height - 1), x]

    return 0.5 * (float(next_value) - float(prev_value))


@cuda.jit('float32(uint8[:,:], float32, float32, int32, int32)', device=True)
def interpolate_bilinear(image: DeviceNDArray, x: float, y: float, height: int, width: int) -> float:
    x0 = int(x // 1)
    y0 = int(y // 1)
    x1 = x0 + 1
    y1 = y0 + 1

    # Avoid pixels outside sensor grid
    if (x0 < 0) or (y0 < 0) or (x1 >= width) or (y1 >= height):
        return np.nan

    x0_weight = x - x0
    x1_weight = x1 - x
    y0_weight = y - y0
    y1_weight = y1 - y

    w00 = x0_weight * y0_weight
    w01 = x0_weight * y1_weight
    w10 = x1_weight * y0_weight
    w11 = x1_weight * y1_weight

    interpolated_value = (
        (w00 * image[y0, x0] + w01 * image[y1, x0] + w10 * image[y0, x1] + w11 * image[y1, x1]) /
        (w00 + w01 + w10 + w11)
    )

    return interpolated_value
    

@cuda.jit('void(uint8[:,:], uint8[:,:], uint16[:,:], float32[:,:], float32[:], float32, float32, float32, float32, float32, boolean[:,:], float32[:,:], float32[:,:], int32, int32)')
def residuals_kernel(
    gray_image: DeviceNDArray, gray_image_prev: DeviceNDArray, depth_image_prev: DeviceNDArray, R: DeviceNDArray,
    tvec: DeviceNDArray, fx: float, fy: float, cx: float, cy: float, depth_scale: float, mask: DeviceNDArray,
    residuals: DeviceNDArray, jacobian: DeviceNDArray, height: int, width: int
):
    tx, ty = cuda.grid(2)

    if (ty >= height) or (tx >= width):
        return

    z = depth_image_prev[ty, tx]

    # Deproject image w.r.t first camera pose
    if (z == 0):

        residuals[ty, tx] = 0.0
        mask[ty, tx] = False

        return

    z = z * depth_scale

    x = (tx - cx) * z / fx
    y = (ty - cy) * z / fy

    # Transform point using estimate
    x1 = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + tvec[0]
    y1 = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + tvec[1]
    z1 = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + tvec[2]

    # Compute Jacobian
    tid = ty * width + tx

    # Compute image gradients
    # NOTE: `J_i(w(se3, x)) = [I2x(w(se3, x)), I2y(w(se3, x))].T` can be approximated by `J_i = [I1x(x), I1y(x)].T`
    gradx = compute_gradients(gray_image_prev, tx, ty, height, width, True)
    grady = compute_gradients(gray_image_prev, tx, ty, height, width, False)
    # gradx = 1.0
    # grady = 1.0

    jacobian[tid, 0] = fx * gradx / z1
    jacobian[tid, 1] = fy * grady / z1
    jacobian[tid, 2] = -fx * gradx * x1 / (z1 ** 2) - fy * grady * y1 / (z1 ** 2)
    jacobian[tid, 3] = -fx * gradx * x1 * y1 / (z1 ** 2) - fy * grady * ((y1 ** 2/z1 ** 2) + 1)
    jacobian[tid, 4] = fx * gradx * (x1 ** 2/(z1 ** 2) + 1) + fy * grady * x1 * y1 / (z1 ** 2)
    jacobian[tid, 5] = - fx * gradx * y1 / z1 + fy * grady * x1 / z1
    
    # Deproject to second sensor plane
    warped_x = fx * x1 / z1 + cx
    warped_y = fy * y1 / z1 + cy

    # Interpolate value for I2
    interpolated_intensity = interpolate_bilinear(gray_image, warped_x, warped_y, height, width)

    if math.isnan(interpolated_intensity):

        residuals[ty, tx] = 0.0
        mask[ty, tx] = False

        return
    
    residuals[ty, tx] = interpolated_intensity - gray_image_prev[ty, tx]
    mask[ty, tx] = True
    
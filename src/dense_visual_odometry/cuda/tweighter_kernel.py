from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np


@cuda.jit("void(float32[:,:], boolean[:,:], float32[:,:], float32, float32, float32, int32, int32, int32, int32)")
def weighting_kernel(
    residuals: DeviceNDArray, mask: DeviceNDArray, weights: DeviceNDArray, dof: float, init_sigma: float,
    tolerance: float, max_iterations: int, N: int, height: int, width: int
):

    # Estimate scale for the T-Distribution

    scale_computation_done = cuda.shared.array(shape=(1), dtype=np.uint8)
    sigma_squared = cuda.shared.array(shape=(1), dtype=np.float32)

    tx, ty = cuda.grid(2)
    tid = ty * width + tx

    if tid == 0:
        last_lambda = 1 / (init_sigma ** 2)
        scale_computation_done[0] = 0

    cuda.syncthreads()

    residual_squared = residuals[ty, tx] ** 2
    k = 1 / float(N)

    for _ in range(max_iterations):

        sigma_squared_i = k * (residual_squared * ((dof + 1) / (dof + residual_squared / sigma_squared[0])))

        if mask[ty, tx]:
            cuda.atomic.add(sigma_squared, 0, sigma_squared_i)

        # First thread is in charge of evaluating termination condition
        if tid == 0:
            curr_lambda = 1 / sigma_squared[0]

            if abs(curr_lambda - last_lambda) < tolerance:
                scale_computation_done[0] = 1

            last_lambda = curr_lambda

        cuda.syncthreads()

        if scale_computation_done[0] > 0:
            break

    # Compute weights
    if mask[ty, tx]:
        weights[ty, tx] = (dof + 1) / (dof + residual_squared / sigma_squared[0])

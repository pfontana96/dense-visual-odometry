from __future__ import division
from numba import cuda, float32

from dense_visual_odometry.cuda import CUDA_BLOCKSIZE


@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
def fast_matmul_kernel(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(CUDA_BLOCKSIZE, CUDA_BLOCKSIZE), dtype=float32)
    sB = cuda.shared.array(shape=(CUDA_BLOCKSIZE, CUDA_BLOCKSIZE), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / CUDA_BLOCKSIZE)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * CUDA_BLOCKSIZE]
        sB[tx, ty] = B[tx + i * CUDA_BLOCKSIZE, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(CUDA_BLOCKSIZE):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

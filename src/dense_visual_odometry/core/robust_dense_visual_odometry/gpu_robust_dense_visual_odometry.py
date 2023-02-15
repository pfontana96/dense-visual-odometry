from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import cuda

from dense_visual_odometry.core.robust_dense_visual_odometry.base_robust_dvo import BaseRobustDVO
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.utils.lie_algebra import Se3
from dense_visual_odometry.utils.image_pyramid import ImagePyramidGPU
from dense_visual_odometry.cuda import CUDA_BLOCKSIZE, residuals_kernel, UnifiedMemoryArray


class RobustDVOGPU(BaseRobustDVO):

    def __init__(
        self, camera_model: RGBDCameraModel, initial_pose: Se3, height: int, width: int, levels: int,
        use_weighter: bool = False, max_increased_steps_allowed: int = 0, sigma: float = None, tolerance: float = 1e-6,
        max_iterations: int = 100, approximate_image2_gradient: bool = False
    ):
        super(RobustDVOGPU, self).__init__(
            camera_model=camera_model, initial_pose=initial_pose, levels=levels, use_weighter=use_weighter,
            max_increased_steps_allowed=max_increased_steps_allowed, sigma=sigma, tolerance=tolerance,
            max_iterations=max_iterations, approximate_image2_gradient=approximate_image2_gradient
        )

        # Allocate and define pointers for image pyramids
        dummy_gray_image = np.zeros((height, width), dtype=np.uint8, order="C")
        dummy_depth_image = np.zeros((height, width), dtype=np.uint16, order="C")

        self._curr_gray_image_pyr = ImagePyramidGPU(image=dummy_gray_image, levels=self._levels, dtype=np.uint8)
        self._prev_gray_image_pyr = ImagePyramidGPU(image=dummy_gray_image, levels=self._levels, dtype=np.uint8)

        self._curr_depth_image_pyr = ImagePyramidGPU(image=dummy_depth_image, levels=self._levels, dtype=np.uint16)
        self._prev_depth_image_pyr = ImagePyramidGPU(image=dummy_depth_image, levels=self._levels, dtype=np.uint16)

        # Allocate memory for estimate
        self._R_uvm = UnifiedMemoryArray((3, 3), np.float32)
        self._tvec_uvm = UnifiedMemoryArray((3,), np.float32)

        # Memory buffers for residuals, jacobian and mask that should be allocated
        self._residuals_uvm = None
        self._mask_uvm = None
        self._jacobian_uvm = None

        # CUDA blockdim
        self._block_dim = (CUDA_BLOCKSIZE, CUDA_BLOCKSIZE)

    def _build_pyramids(
        self, gray_image: npt.NDArray[np.uint8], depth_image: npt.NDArray[np.uint16]
    ):
        """Method resposible for setting `self._curr_gray_image_pyr` and `self._curr_depth_image_pyr`
        """
        self._prev_gray_image_pyr.update(self._gray_image_prev)
        self._prev_depth_image_pyr.update(self._depth_image_prev)
        self._curr_gray_image_pyr.update(gray_image)
        self._curr_depth_image_pyr.update(depth_image)

    def _setup(self, level: int):
        height, width = self._prev_gray_image_pyr.at(level, "cpu").shape
        self._residuals_uvm = UnifiedMemoryArray((height, width), np.float32)
        self._mask_uvm = UnifiedMemoryArray((height, width), bool)
        self._jacobian_uvm = UnifiedMemoryArray((height * width, 6), np.float32)

    def _cleanup(self):
        self._residuals_uvm = None
        self._mask_uvm = None
        self._jacobian_uvm = None

    def compute_residuals_and_jacobian(
        self, estimate: Se3, level: int = 0
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool8]]:
        """Computes residuals of warping pixels in `gray_image_prev` onto `gray_image` by using depth
        information available in `depth_image_prev` and a given estimate of the transform between the
        frames (`estimate`). It also computes the Jacobian of the residuals with respect to the
        parameters of the transform in `estimate`.

        Parameters
        ----------
        estimate : Se3
            Estimate between previous frame and current one.
        level : int, optional
            Image pyramid level to process, by default 0.

        Returns
        -------
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool8]]
            Tuple containing residuals array (Nx1), jacobian array (Nx6) and valid pixels mask (same shape
            as `gray_image`) where N is the number of valid pixels (the sum of ones in valid pixels mask)
        """

        # Define CUDA grid dim
        height, width = self._curr_gray_image_pyr.at(level).shape

        grid_dim = (
            int((width + (CUDA_BLOCKSIZE - 1)) // CUDA_BLOCKSIZE),
            int((height + (CUDA_BLOCKSIZE - 1)) // CUDA_BLOCKSIZE)
        )

        # Compute residuals
        T = np.ascontiguousarray(estimate.exp())

        # Update data from estimate to gpu buffers (same as cpu as we're using UVM)
        self._R_uvm.get("cpu")[...] = T[:3, :3]
        self._tvec_uvm.get("cpu")[...] = T[:3, 3]

        intrinsics = self._camera_model.at(-level)

        residuals_kernel[grid_dim, self._block_dim](
            self._curr_gray_image_pyr.at(level, "gpu"), self._prev_gray_image_pyr.at(level, "gpu"),
            self._prev_depth_image_pyr.at(level, "gpu"), self._R_uvm.get("gpu"), self._tvec_uvm.get("gpu"),
            intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2], self._camera_model.depth_scale,
            self._mask_uvm.get("gpu"), self._residuals_uvm.get("gpu"), self._jacobian_uvm.get("gpu"), height, width
        )

        cuda.synchronize()

        mask = self._mask_uvm.get("cpu")
        residuals = self._residuals_uvm.get("cpu")[mask].reshape(-1, 1)
        jacobian = self._jacobian_uvm.get("cpu")[mask.reshape(-1)]

        return residuals, jacobian, mask

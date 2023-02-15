import logging
from typing import Union
from pathlib import Path

from kornia.feature import LoFTR
import torch
import numpy as np
import cv2

from dense_visual_odometry.core.base_dense_visual_odometry import BaseDenseVisualOdometry, DVOError
from dense_visual_odometry.utils.lie_algebra import Se3, So3
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.utils.match_filtering import RANSAC
from dense_visual_odometry.utils.transform import (
    find_rigid_body_transform_from_pointclouds_SVD1,
    find_rigid_body_transform_from_pointclouds_quat
)


logger = logging.getLogger(__name__)


class LoFTRDVO(BaseDenseVisualOdometry):
    def __init__(
        self, weights_path: Union[str, Path], camera_model: RGBDCameraModel, initial_pose: Se3,
        use_gpu: bool = False, model_config: dict = None, debug_dir: Union[str, Path] = None, use_ransac: bool = False,
        confidence_threshold: float = None, rmse_threshold: float = None, ransac_config: dict = {},
        min_number_of_matches: int = None, max_increased_steps_allowed: int = 0, sigma: float = 1.0
    ):
        super(LoFTRDVO, self).__init__(camera_model=camera_model, initial_pose=initial_pose)
        weights_path = Path(weights_path).resolve()

        if not weights_path.exists():
            raise FileNotFoundError("Could not find LoFTR weights at '{}'".format(str(weights_path)))

        if not weights_path.suffix == ".ckpt":
            raise ValueError(
                "Expected LoFTR weights to be in checkpoint format ('.ckpt'), got '{}' instead".format(
                    weights_path.suffix
                ))

        if model_config is None:
            self._model = LoFTR()

        else:
            self._model = LoFTR(config=model_config)

        # Load weights
        self._model.load_state_dict(torch.load(weights_path)["state_dict"])
        self._model = self._model.eval()

        self._use_gpu = use_gpu
        if self._use_gpu:
            self._model = self._model.cuda()

        self._debug_dir = debug_dir
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        self._use_ransac = use_ransac
        self._ransac_config = LoFTRDVO.load_ransac_config_with_defaults(config=ransac_config)
        if use_ransac:
            self._ransac = RANSAC(
                model=find_rigid_body_transform_from_pointclouds_quat, loss=LoFTRDVO._euclidean_distance_loss,
                metric=LoFTRDVO._rmse, dof=6
            )

        self._confidence_threshold = confidence_threshold
        self._rmse_threshold = rmse_threshold if rmse_threshold is not None else np.finfo("float32").max

        self._min_nb_matches = min_number_of_matches
        self._max_increased_steps_allowed = max_increased_steps_allowed

        self._sigma = sigma

    def find_optimal_transformation(
        self, gray_image_prev: np.ndarray, gray_image: np.ndarray, depth_image_prev: np.ndarray,
        depth_image: np.ndarray, ds_factor_src: int = None, ds_factor_dst: int = None, level: int = 0
    ):
        # Find matches
        curr_pc, prev_pc, weights = self._find_3d_matches(
            gray_image_prev=gray_image_prev, gray_image=gray_image, depth_image_prev=depth_image_prev,
            depth_image=depth_image, ds_factor_src=ds_factor_src, ds_factor_dst=ds_factor_dst, level=level
        )

        # Compute transform (function take pointclouds with shape (3, N))
        if self._use_ransac:
            T, _, error = self._ransac(
                x=prev_pc[:3, :], y=curr_pc[:3, :], weights=weights, threshold=self._ransac_config["threshold"],
                min_count=self._ransac_config["min_count"], max_iter=self._ransac_config["max_iter"]
            )

            if T is None:
                raise DVOError("RANSAC could not estimate a valid model for this frame")

        else:
            try:
                T = find_rigid_body_transform_from_pointclouds_SVD1(
                    src_pc=prev_pc[:3, :], dst_pc=curr_pc[:3, :], weights=weights
                )
                residuals = self._euclidean_distance_loss(src_pc=prev_pc[:3, :], dst_pc=curr_pc[:3, :], T=T)
                error = self._rmse(residuals)

            except Exception as e:
                raise DVOError("Could not estimate valid model for this frame, got '{}'".format(e))

        return Se3(So3(T[:3, :3]), T[:3, 3].reshape(3, 1)), error

    def match_images(
        self, src_image: np.ndarray, dst_image: np.ndarray, ds_factor_src: int = None, ds_factor_dst: int = None
    ):
        original_shape_src = src_image.shape
        original_shape_dst = dst_image.shape

        src_image = LoFTRDVO._prepare_input(image=src_image, downsampling_factor=ds_factor_src)
        dst_image = LoFTRDVO._prepare_input(image=dst_image, downsampling_factor=ds_factor_dst)

        src_tensor = torch.from_numpy(src_image[None, None, ...])
        dst_tensor = torch.from_numpy(dst_image[None, None, ...])

        if self._use_gpu:
            src_tensor = src_tensor.cuda()
            dst_tensor = dst_tensor.cuda()

        batch = {
            "image0": src_tensor,
            "image1": dst_tensor
        }

        with torch.no_grad():
            self._model(batch)

        src_keypoints = batch["mkpts0_f"].cpu().numpy()
        dst_keypoints = batch["mkpts1_f"].cpu().numpy()
        scores = batch["mconf"].cpu().numpy()

        if ds_factor_src is not None:
            src_keypoints = LoFTRDVO._upsample_points(
                points=src_keypoints, original_shape=original_shape_src, downsampled_shape=src_image.shape
            )

        if ds_factor_dst is not None:
            dst_keypoints = LoFTRDVO._upsample_points(
                points=dst_keypoints, original_shape=original_shape_dst, downsampled_shape=dst_image.shape
            )

        return src_keypoints.astype(int), dst_keypoints.astype(int), scores

    @staticmethod
    def _prepare_input(image: np.ndarray, downsampling_factor: int = None):

        if downsampling_factor is not None:
            # Get closest shape divisable by 8
            output_shape = (
                int(np.round(image.shape[0] // downsampling_factor / 8) * 8),
                int(np.round(image.shape[1] // downsampling_factor / 8) * 8)
            )

            image = cv2.resize(image, (output_shape[1], output_shape[0]))

        # Equalize image
        image_equalized = cv2.equalizeHist(image) / 255.0

        return image_equalized.astype(np.float32)

    @staticmethod
    def _upsample_points(points: np.ndarray, original_shape: tuple, downsampled_shape: tuple):
        return points * (np.array(original_shape) / np.array(downsampled_shape))

    @staticmethod
    def _euclidean_distance_loss(src_pc: np.ndarray, dst_pc: np.ndarray, T: np.ndarray, weights: np.ndarray = None):

        residuals = np.linalg.norm(dst_pc - (np.dot(T[:3, :3], src_pc) + T[:3, 3].reshape(-1, 1)), axis=0)

        if weights is not None:
            assert weights.shape == residuals.shape, "Expected 'weights' to have shape '{}', got '{}' instead".format(
                residuals.shape, weights.shape
            )
            residuals *= weights

        return residuals

    @staticmethod
    def _mahalanobis_distance_loss(src_pc: np.ndarray, dst_pc: np.ndarray, T: np.ndarray, weights: np.ndarray = None):
        sigma = 0.05
        residuals = np.linalg.norm(dst_pc - (np.dot(T[:3, :3], src_pc) + T[:3, 3].reshape(-1, 1)), axis=0) * (1 / sigma)

        if weights is not None:
            assert weights.shape == residuals.shape, "Expected 'weights' to have shape '{}', got '{}' instead".format(
                residuals.shape, weights.shape
            )
            residuals *= weights

        return residuals

    @staticmethod
    def _rmse(losses: np.ndarray):
        return np.linalg.norm(losses) / np.sqrt(len(losses))

    def _step(
        self, gray_image: np.ndarray, depth_image: np.ndarray, ds_factor_src: int = None, ds_factor_dst: int = None,
        init_guess: np.ndarray = np.zeros((6, 1), dtype=np.float32)
    ):
        result = None

        try:
            estimated_pose, error = self.find_optimal_transformation(
                gray_image_prev=self._gray_image_prev, gray_image=gray_image, depth_image_prev=self._depth_image_prev,
                depth_image=depth_image, ds_factor_src=ds_factor_src, ds_factor_dst=ds_factor_dst
            )

            if error <= self._rmse_threshold:
                result = estimated_pose
                # result = xi
                logger.debug("RMSE error (dst_pc - T * src_pc): '{}'".format(error))

            else:
                logger.warning("RMSE error (dst_pc - T * src_pc): '{}' higher than threshold '{}'".format(
                    error, self._rmse_threshold
                ))

        except DVOError as e:
            logger.warning(e)

        return result

    def _find_3d_matches(
        self, gray_image_prev: np.ndarray, gray_image: np.ndarray, depth_image_prev: np.ndarray,
        depth_image: np.ndarray, ds_factor_src: int = None, ds_factor_dst: int = None, level: int = 0
    ):
        # Match images
        prev_kps, curr_kps, scores = self.match_images(
            src_image=gray_image_prev, dst_image=gray_image, ds_factor_src=ds_factor_src, ds_factor_dst=ds_factor_dst
        )

        # Filter matches by score
        if self._confidence_threshold is not None:
            mask = scores >= self._confidence_threshold

            prev_kps = prev_kps[mask]
            curr_kps = curr_kps[mask]
            scores = scores[mask]

        if self._min_nb_matches is not None:
            if prev_kps.shape[0] < self._min_nb_matches:
                raise DVOError("Expected to find at least '{}' matches with confidence >= '{}', found '{}'".format(
                    self._min_nb_matches, self._confidence_threshold, prev_kps.shape[0]
                ))

        # Filter keypoints on where either src or dst have no valid depth data
        depth_valid_kps_mask = np.logical_and(
            depth_image_prev[prev_kps[:, 1], prev_kps[:, 0]] != 0.0,
            depth_image[curr_kps[:, 1], curr_kps[:, 0]] != 0.0
        )

        prev_kps = prev_kps[depth_valid_kps_mask]
        curr_kps = curr_kps[depth_valid_kps_mask]
        scores = scores[depth_valid_kps_mask]

        # Consider only LoFTR features for computing transformation
        prev_pixel_mask = np.zeros_like(depth_image_prev, dtype=bool)
        prev_pixel_mask[prev_kps[:, 1], prev_kps[:, 0]] = True

        curr_pixel_mask = np.zeros_like(depth_image, dtype=bool)
        curr_pixel_mask[curr_kps[:, 1], curr_kps[:, 0]] = True

        prev_pc = self._camera_model.deproject_unsafe(depth_image=depth_image_prev, mask=prev_pixel_mask, level=level)
        curr_pc = self._camera_model.deproject_unsafe(depth_image=depth_image, mask=curr_pixel_mask, level=level)

        return prev_pc, curr_pc, scores

    @staticmethod
    def load_ransac_config_with_defaults(config: dict):

        result = {}
        result["min_count"] = config.get("min_count", 10)
        result["threshold"] = config.get("threshold", 0.1)
        result["max_iter"] = config.get("max_iter", 100)

        return result

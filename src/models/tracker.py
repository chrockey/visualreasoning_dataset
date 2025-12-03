from typing import Any, Dict, Optional, Literal
from abc import ABC, abstractmethod

import numpy as np
import torch

from .cotracker import CoTracker
from .sam2 import SAM2VideoPredictorWrapper


class BaseTracker(ABC):
    """Base class for keypoint tracking."""
    
    @abstractmethod
    def track(
        self,
        video: np.ndarray,
        keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track keypoints across video frames.
        
        Args:
            video: Array of shape (T, H, W, 3) - video frames
            keypoints: Array of shape (1, N, 2) or (N, 2) - initial keypoint coordinates
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility)
            - tracked_keypoints: Array of shape (B, T, N, 2) - tracked coordinates over time
            - tracked_visibility: Array of shape (B, T, N) - visibility scores per track
        """
        pass
    
    @abstractmethod
    def track_from_masks(
        self,
        video: np.ndarray,
        masks: np.ndarray,
        masks_frame_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track objects from masks across video frames.
        
        Args:
            video: Array of shape (T, H, W, 3) - video frames
            masks: Binary masks as numpy array (n, H, W) - initial masks
            masks_frame_idx: Frame index where masks are located (default: 0)
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility)
            - tracked_keypoints: Array of shape (B, T, N, 2) - tracked coordinates over time
            - tracked_visibility: Array of shape (B, T, N) - visibility scores per track
            Note: N = n * K where K is keypoints per mask (3 for CoTracker, 1 for SAM2)
        """
        pass
    
    @abstractmethod
    def extract_keypoints_from_masks(self, masks: np.ndarray) -> np.ndarray:
        """Extract keypoints from masks.
        
        Args:
            masks: Binary masks as numpy array (n, H, W)
        
        Returns:
            Keypoints as numpy array (n, K, 2) where each mask has K keypoints (x, y)
            K may vary by tracker type (e.g., 3 for CoTracker, 1 for SAM2)
        """
        pass


class CoTrackerWrapper(BaseTracker):
    """Wrapper for CoTracker that conforms to the unified tracking interface."""
    
    def __init__(
        self,
        model_id: str = "facebookresearch/co-tracker:cotracker3_offline",
        grid_size: int = 0,
        grid_query_frame: int = 0,
    ):
        """Initialize CoTracker wrapper.
        
        Args:
            model_id: Model ID for CoTracker
            grid_size: Grid size for dense tracking (0 to disable)
            grid_query_frame: Frame index for grid queries
        """
        self.tracker = CoTracker(model_id)
        self.grid_size = grid_size
        self.grid_query_frame = grid_query_frame
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def track(
        self,
        video: np.ndarray,
        keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track keypoints using CoTracker.
        
        Args:
            video: Array of shape (T, H, W, 3) - video frames
            keypoints: Array of shape (1, N, 2) or (N, 2) - initial keypoint coordinates
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility)
            - tracked_keypoints: Array of shape (B, T, N, 2)
            - tracked_visibility: Array of shape (B, T, N)
        """
        # Normalize keypoints shape
        if keypoints.ndim == 2:
            keypoints = keypoints[np.newaxis, ...]  # (1, N, 2)
        
        # Convert to torch tensor if needed
        if isinstance(keypoints, np.ndarray):
            keypoints_tensor = torch.from_numpy(keypoints).float()
        else:
            keypoints_tensor = keypoints
        
        # Track using CoTracker
        tracked_keypoints, tracked_visibility = self.tracker(
            video=video,
            queries=keypoints_tensor.to(self.device),
            grid_size=self.grid_size,
            grid_query_frame=self.grid_query_frame,
        )
        
        # Convert to numpy if needed
        if isinstance(tracked_keypoints, torch.Tensor):
            tracked_keypoints = tracked_keypoints.cpu().numpy()
        if isinstance(tracked_visibility, torch.Tensor):
            tracked_visibility = tracked_visibility.cpu().numpy()
        
        return tracked_keypoints, tracked_visibility
    
    def track_from_masks(
        self,
        video: np.ndarray,
        masks: np.ndarray,
        masks_frame_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track objects from masks using CoTracker.
        
        Args:
            video: Array of shape (T, H, W, 3) - video frames
            masks: Binary masks as numpy array (n, H, W) - initial masks
            masks_frame_idx: Frame index where masks are located (default: 0)
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility)
            - tracked_keypoints: Array of shape (B, T, N, 2) where N = n * 3
            - tracked_visibility: Array of shape (B, T, N)
        """
        # Extract keypoints from masks
        keypoints = self.extract_keypoints_from_masks(masks)  # (n, 3, 2)
        # Reshape to (1, n*3, 2) for tracking
        keypoints_flat = keypoints.reshape(1, -1, 2)
        return self.track(video, keypoints_flat)
    
    def extract_keypoints_from_masks(self, masks: np.ndarray) -> np.ndarray:
        """Extract keypoints from masks using CoTracker's method (3 keypoints per mask).
        
        Args:
            masks: Binary masks as numpy array (n, H, W)
        
        Returns:
            Keypoints as numpy array (n, 3, 2) where each mask has 3 keypoints (x, y)
        """
        return CoTracker.extract_keypoints_from_masks(masks)


class SAM2TrackerWrapper(BaseTracker):
    """Wrapper for SAM2 VideoPredictor that uses mask propagation for tracking."""
    
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        reverse: bool = False,
        **kwargs,
    ):
        """Initialize SAM2 tracker wrapper.
        
        Args:
            model_id: HuggingFace model ID for SAM2
            offload_video_to_cpu: Whether to offload video frames to CPU memory
            offload_state_to_cpu: Whether to offload inference state to CPU memory
            reverse: Whether to use reverse propagation
            **kwargs: Additional arguments passed to SAM2VideoPredictorWrapper
        """
        self.predictor = SAM2VideoPredictorWrapper(
            model_id=model_id,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            **kwargs,
        )
        self.reverse = reverse
    
    def _extract_tracks_from_propagated_masks(
        self,
        video_segments: dict[int, np.ndarray],
        num_frames: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Extract tracks from propagated masks (common logic for track and track_from_masks).
        
        Args:
            video_segments: Dictionary mapping frame indices to propagated masks
            num_frames: Total number of frames in video
        
        Returns:
            Tuple of (tracks, visibility) where each is a list of arrays
        """
        tracks = []
        visibility = []
        
        for frame_idx in range(num_frames):
            if frame_idx in video_segments:
                mask = np.squeeze(video_segments[frame_idx])  # (H, W) binary mask
                
                if mask.sum() > 0:
                    # Extract centroid from mask
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
                        centroid = points.mean(axis=0)  # (2,)
                        
                        tracks.append(centroid)
                        visibility.append(1.0)
                    else:
                        # Empty mask
                        if len(tracks) > 0:
                            tracks.append(tracks[-1].copy())
                        else:
                            tracks.append(np.array([0.0, 0.0]))
                        visibility.append(0.0)
                else:
                    # Empty mask
                    if len(tracks) > 0:
                        tracks.append(tracks[-1].copy())
                    else:
                        tracks.append(np.array([0.0, 0.0]))
                    visibility.append(0.0)
            else:
                # Frame not in segments
                if len(tracks) > 0:
                    tracks.append(tracks[-1].copy())
                else:
                    tracks.append(np.array([0.0, 0.0]))
                visibility.append(0.0)
        
        return tracks, visibility
    
    def _stack_tracks(
        self,
        all_tracks: list[list[np.ndarray]],
        all_visibility: list[list[np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack tracks from multiple objects into arrays.
        
        Args:
            all_tracks: List of track lists, one per object
            all_visibility: List of visibility lists, one per object
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility) with shape (1, T, N, 2) and (1, T, N)
        """
        # Stack: (N, T, 2) -> (T, N, 2)
        tracked_keypoints_array = np.stack(
            [np.stack(tracks) for tracks in all_tracks], axis=1
        )  # (T, N, 2)
        tracked_visibility_array = np.stack(
            [np.array(vis) for vis in all_visibility], axis=1
        )  # (T, N)
        
        # Add batch dimension: (1, T, N, 2) and (1, T, N)
        tracked_keypoints_array = tracked_keypoints_array[np.newaxis, ...]
        tracked_visibility_array = tracked_visibility_array[np.newaxis, ...]
        
        return tracked_keypoints_array, tracked_visibility_array
    
    def track(
        self,
        video: np.ndarray,
        keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track keypoints using SAM2 mask propagation.
        
        This method propagates masks from initial keypoints and extracts
        keypoints from each frame's propagated mask to create tracks.
        
        Args:
            video: Array of shape (T, H, W, 3) - video frames
            keypoints: Array of shape (1, N, 2) or (N, 2) - initial keypoint coordinates
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility)
            - tracked_keypoints: Array of shape (B, T, N, 2)
            - tracked_visibility: Array of shape (B, T, N)
        """
        # Normalize keypoints shape
        if keypoints.ndim == 3:
            keypoints = keypoints[0]  # (N, 2)
        
        T, H, W, C = video.shape
        num_keypoints = keypoints.shape[0]
        points_frame_idx = 0
        
        # Track each keypoint individually using mask propagation
        # Each keypoint gets its own mask which we track across frames
        all_keypoint_tracks = []
        all_keypoint_visibility = []
        
        for kp_idx in range(num_keypoints):
            single_keypoint = keypoints[kp_idx : kp_idx + 1]  # (1, 2)
            
            # Propagate mask from this keypoint
            video_segments = self.predictor.video_inference(
                video_frames=video,
                points=single_keypoint,
                prompt_frame_idx=points_frame_idx,
                reverse=self.reverse,
            )
            
            # Extract tracks from propagated masks
            kp_tracks, kp_visibility = self._extract_tracks_from_propagated_masks(
                video_segments, T
            )
            
            all_keypoint_tracks.append(kp_tracks)
            all_keypoint_visibility.append(kp_visibility)
        
        return self._stack_tracks(all_keypoint_tracks, all_keypoint_visibility)
    
    def track_from_masks(
        self,
        video: np.ndarray,
        masks: np.ndarray,
        masks_frame_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track objects from masks using SAM2 mask propagation.
        
        This method directly uses masks for propagation, which is more efficient
        than extracting keypoints first.
        
        Args:
            video: Array of shape (T, H, W, 3) - video frames
            masks: Binary masks as numpy array (n, H, W) - initial masks
            masks_frame_idx: Frame index where masks are located (default: 0)
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility)
            - tracked_keypoints: Array of shape (B, T, N, 2) where N = n (1 keypoint per mask)
            - tracked_visibility: Array of shape (B, T, N)
        """
        T, H, W, C = video.shape
        num_masks = masks.shape[0]
        
        # Track each mask individually using mask propagation
        all_mask_tracks = []
        all_mask_visibility = []
        
        for mask_idx in range(num_masks):
            mask = masks[mask_idx]  # (H, W) binary mask
            
            # Propagate mask through video
            video_segments = self._propagate_mask(
                video_frames=video,
                mask=mask,
                mask_frame_idx=masks_frame_idx,
            )
            
            # Extract tracks from propagated masks
            mask_tracks, mask_visibility = self._extract_tracks_from_propagated_masks(
                video_segments, T
            )
            
            all_mask_tracks.append(mask_tracks)
            all_mask_visibility.append(mask_visibility)
        
        return self._stack_tracks(all_mask_tracks, all_mask_visibility)
    
    def _propagate_mask(
        self,
        video_frames: np.ndarray,
        mask: np.ndarray,
        mask_frame_idx: int = 0,
    ) -> dict[int, np.ndarray]:
        """Propagate a single mask through video using SAM2.
        
        Args:
            video_frames: Array of shape (T, H, W, 3) - video frames
            mask: Binary mask as numpy array (H, W) - initial mask
            mask_frame_idx: Frame index where mask is located (default: 0)
        
        Returns:
            Dictionary mapping frame indices to propagated masks
        """
        # Use SAM2VideoPredictorWrapper's mask-based inference
        return self.predictor.video_inference_with_mask(
            video_frames=video_frames,
            mask=mask,
            mask_frame_idx=mask_frame_idx,
            reverse=self.reverse,
        )
    
    def extract_keypoints_from_masks(self, masks: np.ndarray) -> np.ndarray:
        """Extract keypoints from masks using centroid (1 keypoint per mask for SAM2).
        
        Args:
            masks: Binary masks as numpy array (n, H, W)
        
        Returns:
            Keypoints as numpy array (n, 1, 2) where each mask has 1 keypoint (centroid) (x, y)
        """
        keypoints_list = []
        for i in range(masks.shape[0]):
            mask = masks[i]
            if mask.sum() > 0:
                # Extract centroid
                y_coords, x_coords = np.where(mask)
                points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
                centroid = points.mean(axis=0)  # (2,)
            else:
                # Empty mask - return zero
                centroid = np.array([0.0, 0.0], dtype=np.float32)
            
            # Reshape to (1, 2) to match CoTracker's format (n, K, 2)
            keypoints_list.append(centroid[np.newaxis, :])
        
        return np.stack(keypoints_list, axis=0)  # (n, 1, 2)


class KeypointTracker:
    """Unified keypoint tracker that supports multiple tracking backends."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        tracker_type: Optional[Literal["cotracker", "sam2"]] = None,
    ):
        """Initialize keypoint tracker.
        
        Args:
            config: Configuration dictionary
            tracker_type: Explicit tracker type override. If None, uses config["keypoint_tracker"]["model_id"]
        """
        self.config = config
        
        # Determine tracker type
        if tracker_type is None:
            keypoint_tracker_config = config.get("keypoint_tracker", {})
            if isinstance(keypoint_tracker_config, dict):
                tracker_type = keypoint_tracker_config.get("model_id", "cotracker")
            else:
                tracker_type = "cotracker"
        
        if tracker_type == "cotracker":
            cotracker_config = config.get("cotracker", {})
            self.tracker = CoTrackerWrapper(
                model_id=cotracker_config.get("model_id", "facebookresearch/co-tracker:cotracker3_offline"),
                grid_size=cotracker_config.get("grid_size", 0),
                grid_query_frame=cotracker_config.get("grid_query_frame", 0),
            )
        elif tracker_type == "sam2":
            sam2_config = config.get("sam2", {})
            self.tracker = SAM2TrackerWrapper(
                model_id=sam2_config.get("model_id", "facebook/sam2-hiera-large"),
                offload_video_to_cpu=sam2_config.get("offload_video_to_cpu", False),
                offload_state_to_cpu=sam2_config.get("offload_state_to_cpu", False),
                reverse=sam2_config.get("reverse", False),
            )
        else:
            raise ValueError(
                f"Invalid keypoint tracker model: {tracker_type}. "
                f"Supported types: 'cotracker', 'sam2'"
            )
        
        self.tracker_type = tracker_type
    
    def __call__(
        self,
        video: np.ndarray,
        keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track keypoints across video frames.
        
        Args:
            video: Array of shape (T, H, W, 3) - video frames
            keypoints: Array of shape (1, N, 2) or (N, 2) - initial keypoint coordinates
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility)
            - tracked_keypoints: Array of shape (B, T, N, 2) - tracked coordinates over time
            - tracked_visibility: Array of shape (B, T, N) - visibility scores per track
        """
        return self.tracker.track(video, keypoints)
    
    def track_from_masks(
        self,
        video: np.ndarray,
        masks: np.ndarray,
        masks_frame_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track objects from masks across video frames.
        
        This is more efficient than extracting keypoints first, especially for SAM2.
        
        Args:
            video: Array of shape (T, H, W, 3) - video frames
            masks: Binary masks as numpy array (n, H, W) - initial masks
            masks_frame_idx: Frame index where masks are located (default: 0)
        
        Returns:
            Tuple of (tracked_keypoints, tracked_visibility)
            - tracked_keypoints: Array of shape (B, T, N, 2) - tracked coordinates over time
            - tracked_visibility: Array of shape (B, T, N) - visibility scores per track
        """
        return self.tracker.track_from_masks(video, masks, masks_frame_idx)
    
    def extract_keypoints_from_masks(self, masks: np.ndarray) -> np.ndarray:
        """Extract keypoints from masks using the appropriate method for the tracker type.
        
        Args:
            masks: Binary masks as numpy array (n, H, W)
        
        Returns:
            Keypoints as numpy array (n, K, 2) where:
            - K=3 for CoTracker (3 keypoints per mask)
            - K=1 for SAM2 (centroid per mask)
        """
        return self.tracker.extract_keypoints_from_masks(masks)


class KeypointFilter:
    """Rule-based filtering system for tracked keypoints."""
    
    def __init__(
        self, 
        traj_top_k: int = 3,
        drop_length_ratio_threshold: float = 0.1,
        drop_use_median: bool = True,
    ):
        """
        Args:
            traj_top_k: Number of top moving keypoints to select
            drop_length_ratio_threshold: Threshold for filtering outliers based on trajectory length.
                Trajectories outside [center * (1 - threshold), center * (1 + threshold)] are removed.
                Default 0.1 means trajectories within 90%-110% of center value are kept.
            drop_use_median: If True, use median as center (robust to outliers).
                       If False, use mean as center.
        """
        self.traj_top_k = traj_top_k
        
        # for dropping outlier trajectories
        self.drop_length_ratio_threshold = drop_length_ratio_threshold
        self.drop_use_median = drop_use_median

    @staticmethod
    def _top_moving_keypoints(
        tracked_keypoints: np.ndarray, 
        tracked_visibility: np.ndarray, 
        top_k: int = 3
    ):
        """Select top-k moving keypoints based on displacement.
        
        Args:
            tracked_keypoints: (B, T, N, 2) trajectory coordinates
            tracked_visibility: (B, T, N) visibility scores
            top_k: Number of top keypoints to select
        
        Returns:
            Tuple of (filtered_keypoints, filtered_visibility, num_kept)
        """
        B, T, N, _ = tracked_keypoints.shape

        if N <= top_k:
            print(f"Number of keypoints ({N}) is less than or equal to top_k ({top_k}). No filtering applied.")
            return tracked_keypoints, tracked_visibility, N

        # Calculate total displacement for each keypoint from start to end frame
        displacements = np.zeros((B, N))

        for b in range(B):
            for i in range(N):
                start_pos = tracked_keypoints[b, 0, i]  # First frame
                end_pos = tracked_keypoints[b, -1, i]   # Last frame
                displacement = np.linalg.norm(end_pos - start_pos)
                displacements[b, i] = displacement

        # Average displacement across batch
        avg_displacements = displacements.mean(axis=0)

        # Get indices of top-k keypoints with largest displacement
        # Use copy() to avoid negative stride issues
        top_indices = np.argsort(avg_displacements)[::-1][:top_k].copy()

        # Filter keypoints and visibility
        filtered_keypoints = tracked_keypoints[:, :, top_indices, :]
        filtered_visibility = tracked_visibility[:, :, top_indices]

        return filtered_keypoints, filtered_visibility, len(top_indices)
    
    def _drop_outlier_trajectories(
        self, 
        tracked_keypoints: np.ndarray, 
        tracked_visibility: np.ndarray
    ):
        """Drop outlier trajectories based on trajectory length using simple ratio-based filtering.
        
        Args:
            tracked_keypoints: (B, T, N, 2) trajectory coordinates
            tracked_visibility: (B, T, N) visibility scores
            
        Returns:
            Tuple of (filtered_keypoints, filtered_visibility, num_kept)
        """
        B, T, N, _ = tracked_keypoints.shape
        
        if N == 0 or N == 1:
            return tracked_keypoints, tracked_visibility, N
        
        # Calculate total trajectory length (cumsum length) for each trajectory
        frame_diffs = tracked_keypoints[:, 1:, :, :] - tracked_keypoints[:, :-1, :, :]
        frame_distances = np.linalg.norm(frame_diffs, axis=3)  # (B, T-1, N)
        # Use cumsum to get cumulative length, then take the last value (total length)
        cumsum_lengths = np.cumsum(frame_distances, axis=1)  # (B, T-1, N)
        total_lengths = cumsum_lengths[:, -1, :].squeeze()  # (N,) - total trajectory length
        
        # Use median as center
        if self.drop_use_median:
            center_length = np.median(total_lengths)
        else:
            center_length = total_lengths.mean()
        
        # Filter trajectories greater than center * (1 - threshold)
        lower_bound = center_length * (1 - self.drop_length_ratio_threshold)
        valid_mask = (total_lengths >= lower_bound)
        valid_indices = np.where(valid_mask)[0]
        
        # Filter keypoints and visibility
        filtered_keypoints = tracked_keypoints[:, :, valid_indices, :]
        filtered_visibility = tracked_visibility[:, :, valid_indices]
        
        return filtered_keypoints, filtered_visibility, len(valid_indices)
        
    def __call__(
        self, 
        tracked_keypoints: np.ndarray, 
        tracked_visibility: np.ndarray
    ):
        """Filter tracked keypoints using rule-based methods.
        
        Args:
            tracked_keypoints: (B, T, N, 2) trajectory coordinates
            tracked_visibility: (B, T, N) visibility scores
        
        Returns:
            Tuple of (filtered_keypoints, filtered_visibility, num_kept)
        """
        # 1. Select top-k moving keypoints
        tracked_keypoints, tracked_visibility, keypoint_num = self._top_moving_keypoints(
            tracked_keypoints, tracked_visibility, top_k=self.traj_top_k
        )
        
        # 2. Drop outlier trajectories from the selected set
        tracked_keypoints, tracked_visibility, keypoint_num = self._drop_outlier_trajectories(
            tracked_keypoints, tracked_visibility
        )
        
        return tracked_keypoints, tracked_visibility, keypoint_num


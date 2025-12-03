from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from .base import BasePipeline, load_config
from ..models.gemma import Gemma
from ..models.grounded_sam import GroundedSAM2
from ..models.tracker import KeypointFilter
from ..models.tracker import KeypointTracker
from ..visualizers.visual_trace import VisualTraceVisualizer

# TODO : Implement the VisualTracePipeline
# 1. Load the Gemma, CoTracker v3, Grounded-SAM2 models
# 2. extract main object from the image using Gemma
# 3. Grounded SAM2 to segment the main object and extract the keypoints
# 4. track the keypoints using CoTracker v3 or SAM2
# 5. filter the keypoints using the confidence score


@dataclass
class TrackingData:
    """Container for tracking results."""
    filtered_keypoints: np.ndarray
    filtered_visibility: np.ndarray
    keypoints_for_viz: np.ndarray
    n_keypoints: int
    masks: np.ndarray
    scores: np.ndarray
    logits: np.ndarray
    boxes: np.ndarray
    labels: list


class VisualTracePipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any], verbose: bool = True, suffix: str = "base"):
        super().__init__(config)
        self.object_extractor = Gemma(config["gemma"]["model_id"])
        
        # Initialize GroundedSam with config
        grounded_sam_config = config.get("grounded_sam", {}).copy()
        self.mask_selection_config = grounded_sam_config.pop("mask_selection", {})
        self.mask_selection_enabled = self.mask_selection_config.get("enabled", False)
        self.mask_selection_top_k = self.mask_selection_config.get("top_k", 1)
        self.grounded_segmenter = GroundedSAM2(**grounded_sam_config)
        
        # Initialize keypoint tracker (supports both CoTracker and SAM2)
        self.keypoint_tracker = KeypointTracker(config)
        self.verbose = verbose
        self.visualizer = VisualTraceVisualizer(
            save_dir=f"viz/visual_trace_{suffix}",
            tracks_leave_trace=-1,
            save_per_mask=False,  # Disable per-mask video saving (all keypoints video always enabled)
        )

        # Initialize keypoint filter
        keypoint_filter_config = config["keypoint_filter"]
        self.keypoint_filter = KeypointFilter(
            traj_top_k=keypoint_filter_config["traj_top_k"],
            drop_length_ratio_threshold=keypoint_filter_config["traj_drop_len_ratio"],
            drop_use_median=keypoint_filter_config["traj_drop_use_median"],
        )

        # Keypoints per mask configuration
        self.keypoints_per_mask = config["cotracker"].get("keypoints_per_mask", 3)

        # Robot gripper configuration
        robot_gripper_config = config.get("robot_gripper", {})
        self.track_robot_gripper = robot_gripper_config.get("enabled", False)
        self.robot_gripper_prompt = robot_gripper_config.get("prompt")

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def _select_top_masks(self, masks, scores, logits, boxes, labels):
        if (
            not self.mask_selection_enabled
            or masks is None
            or masks.shape[0] == 0
            or scores is None
        ):
            return masks, scores, logits, boxes, labels

        scores_array = np.asarray(scores).reshape(-1)
        top_k = self.mask_selection_top_k
        if top_k is None or top_k <= 0:
            return masks, scores, logits, boxes, labels
        top_k = min(top_k, masks.shape[0])
        if top_k >= masks.shape[0]:
            return masks, scores_array, logits, boxes, labels

        top_indices = np.argsort(scores_array)[::-1][:top_k]
        masks = masks[top_indices]
        scores_array = scores_array[top_indices]
        logits = logits[top_indices] if logits is not None else None
        boxes = boxes[top_indices] if boxes is not None else None
        if labels is not None:
            if isinstance(labels, list):
                labels = [labels[i] for i in top_indices]
            else:
                labels = labels[top_indices]
        return masks, scores_array, logits, boxes, labels

    def _segment_and_select_masks(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Segment object and select top masks.
        
        Args:
            image: Input image
            prompt: Text prompt for segmentation
        
        Returns:
            Tuple of (masks, scores, logits, boxes, labels)
        """
        masks, scores, logits, boxes, labels = self.grounded_segmenter(image, prompt)
        masks, scores, logits, boxes, labels = self._select_top_masks(
            masks, scores, logits, boxes, labels
        )
        
        # Normalize outputs
        scores = np.asarray(scores).reshape(-1) if scores is not None else None
        logits = np.asarray(logits)
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = np.squeeze(logits, axis=1)
        
        self.grounded_segmenter.reset_predictor()
        return masks, scores, logits, boxes, labels

    def _track_object_from_masks(
        self,
        video_frames: np.ndarray,
        masks: np.ndarray,
        clip_idx: int,
        object_name: str = "object",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Track an object from masks and return filtered results.
        
        Args:
            video_frames: Video frames array
            masks: Binary masks array
            clip_idx: Clip index for logging
            object_name: Name of the object for logging
        
        Returns:
            Tuple of (filtered_keypoints, filtered_visibility, keypoints_for_viz, num_keypoints)
        """
        # Track from masks
        tracked_keypoints, tracked_visibility = self.keypoint_tracker.track_from_masks(
            video_frames, masks, masks_frame_idx=0
        )
        
        # Extract keypoints for visualization
        keypoints = self.keypoint_tracker.extract_keypoints_from_masks(masks)
        
        # Filter keypoints
        filtered_keypoints, filtered_visibility, n_keypoints = self.keypoint_filter(
            tracked_keypoints, tracked_visibility
        )
        
        if self.verbose:
            print(f"Clip {clip_idx}: {object_name} tracked with {n_keypoints} keypoints")
        
        return filtered_keypoints, filtered_visibility, keypoints, n_keypoints

    def _combine_tracking_results(
        self,
        task_keypoints: np.ndarray,
        task_filtered_keypoints: np.ndarray,
        task_filtered_visibility: np.ndarray,
        task_n_keypoints: int,
        num_task_masks: int,
        points_per_mask: int,
        gripper_keypoints: Optional[np.ndarray] = None,
        gripper_filtered_keypoints: Optional[np.ndarray] = None,
        gripper_filtered_visibility: Optional[np.ndarray] = None,
        gripper_n_keypoints: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Combine task and gripper tracking results.
        
        Args:
            task_keypoints: Task object keypoints for visualization
            task_filtered_keypoints: Filtered task keypoints
            task_filtered_visibility: Filtered task visibility
            task_n_keypoints: Number of task keypoints
            num_task_masks: Number of task masks
            points_per_mask: Number of points per mask
            gripper_keypoints: Optional gripper keypoints for visualization
            gripper_filtered_keypoints: Optional filtered gripper keypoints
            gripper_filtered_visibility: Optional filtered gripper visibility
            gripper_n_keypoints: Optional number of gripper keypoints
        
        Returns:
            Tuple of (combined_keypoints, combined_visibility, all_keypoints, point_to_mask, keypoint_types)
        """
        if gripper_keypoints is None:
            # No gripper, return task results only
            point_to_mask = np.repeat(
                np.arange(num_task_masks)[:, None], points_per_mask, axis=1
            ).reshape(-1)
            return (
                task_filtered_keypoints,
                task_filtered_visibility,
                task_keypoints,
                point_to_mask,
                None,  # keypoint_types
            )
        
        # Combine task and gripper keypoints
        combined_keypoints = np.concatenate(
            [task_filtered_keypoints, gripper_filtered_keypoints], axis=2
        )
        combined_visibility = np.concatenate(
            [task_filtered_visibility, gripper_filtered_visibility], axis=2
        )
        
        # Create keypoint_types array (0 for task, 1 for gripper)
        keypoint_types = np.concatenate([
            np.zeros(task_n_keypoints, dtype=np.int32),
            np.ones(gripper_n_keypoints, dtype=np.int32)
        ])
        
        # Combine keypoints for visualization
        all_keypoints = np.concatenate([task_keypoints, gripper_keypoints], axis=0)
        
        # Build point_to_mask mapping
        task_point_to_mask = np.repeat(
            np.arange(num_task_masks)[:, None], points_per_mask, axis=1
        ).reshape(-1)
        gripper_point_to_mask = np.repeat(
            np.arange(num_task_masks, num_task_masks + gripper_keypoints.shape[0])[:, None],
            points_per_mask,
            axis=1
        ).reshape(-1)
        point_to_mask = np.concatenate([task_point_to_mask, gripper_point_to_mask])
        
        return combined_keypoints, combined_visibility, all_keypoints, point_to_mask, keypoint_types

    def _track_gripper(
        self,
        video_frames: np.ndarray,
        clip_frames: np.ndarray,
        frame_idx: int,
        clip_idx: int,
        gripper_word: str,
    ) -> Optional[TrackingData]:
        """Track robot gripper if enabled.
        
        Args:
            video_frames: Full video frames array (for segmentation)
            clip_frames: Clip frames array (for tracking, should match task object tracking)
            frame_idx: Frame index to segment gripper from (relative to video_frames)
            clip_idx: Clip index for logging
            gripper_word: Text prompt for gripper segmentation
        
        Returns:
            TrackingData if gripper is detected, None otherwise
        """
        if not self.track_robot_gripper:
            return None
        
        gripper_masks, gripper_scores, gripper_logits, gripper_boxes, gripper_labels = \
            self._segment_and_select_masks(video_frames[frame_idx], gripper_word)
        
        if gripper_masks is None or gripper_masks.shape[0] == 0:
            if self.verbose:
                print(f"Clip {clip_idx}: Warning: Robot gripper not detected for this clip")
            return None
        
        gripper_filtered_keypoints, gripper_filtered_visibility, gripper_keypoints, gripper_n_keypoints = \
            self._track_object_from_masks(
                clip_frames, gripper_masks, clip_idx, "Robot gripper"
            )
        
        if self.verbose:
            print(
                f"Clip {clip_idx}: Robot gripper detected with {gripper_keypoints.shape[0]} masks"
            )
        
        return TrackingData(
            filtered_keypoints=gripper_filtered_keypoints,
            filtered_visibility=gripper_filtered_visibility,
            keypoints_for_viz=gripper_keypoints,
            n_keypoints=gripper_n_keypoints,
            masks=gripper_masks,
            scores=gripper_scores,
            logits=gripper_logits,
            boxes=gripper_boxes,
            labels=gripper_labels,
        )

    def process(self, data_dict: Dict[str, Any]):
        descriptions = data_dict["descriptions"]
        video_frames = data_dict["frames"]
        global_frames_array = None

        gripper_word = self.robot_gripper_prompt + "." if self.track_robot_gripper else None

        for clip_idx, description in enumerate(descriptions):
            str_idx, end_idx, task_prompt = description
            word = self.object_extractor(task_prompt).replace("\n", "") + "."
            image = video_frames[str_idx]

            if self.verbose:
                print(f"Clip {clip_idx}: Task prompt: '{task_prompt}'")
                print(f"Clip {clip_idx}: Gemma extracted key object: '{word}'")

            # Segment and track task object
            masks, scores, logits, boxes, labels = self._segment_and_select_masks(image, word)
            
            if masks is None or masks.shape[0] == 0:
                error_msg = (
                    f"[VisualTracePipeline] Grounded SAM2 failed to produce masks "
                    f"for clip {clip_idx} with prompt '{word}'."
                )
                if self.verbose:
                    print(error_msg)
                raise RuntimeError(error_msg)
            
            # Track task object
            task_filtered_keypoints, task_filtered_visibility, task_keypoints, task_n_keypoints = \
                self._track_object_from_masks(
                    video_frames[str_idx:end_idx], masks, clip_idx, "Task object"
                )
            num_task_masks, points_per_mask = task_keypoints.shape[:2]

            # Track gripper if enabled
            gripper_data = self._track_gripper(
                video_frames, video_frames[str_idx:end_idx], str_idx, clip_idx, gripper_word
            ) if gripper_word else None

            # Combine tracking results
            if gripper_data:
                tracked_keypoints, tracked_visibility, all_keypoints, point_to_mask, keypoint_types = \
                    self._combine_tracking_results(
                        task_keypoints,
                        task_filtered_keypoints,
                        task_filtered_visibility,
                        task_n_keypoints,
                        num_task_masks,
                        points_per_mask,
                        gripper_data.keypoints_for_viz,
                        gripper_data.filtered_keypoints,
                        gripper_data.filtered_visibility,
                        gripper_data.n_keypoints,
                    )
                # Combine masks for visualization
                masks = np.concatenate([masks, gripper_data.masks], axis=0)
                scores = np.concatenate([scores, gripper_data.scores], axis=0)
                logits = np.concatenate([logits, gripper_data.logits], axis=0)
                boxes = np.concatenate([boxes, gripper_data.boxes], axis=0)
                labels = np.concatenate([labels, gripper_data.labels], axis=0)
            else:
                tracked_keypoints, tracked_visibility, all_keypoints, point_to_mask, keypoint_types = \
                    self._combine_tracking_results(
                        task_keypoints,
                        task_filtered_keypoints,
                        task_filtered_visibility,
                        task_n_keypoints,
                        num_task_masks,
                        points_per_mask,
                    )

            if self.verbose:
                # Extract data_name from data_dict if available
                data_name = data_dict.get("video_name", data_dict.get("name", None))
                global_frames_array = self.visualizer.visualize_tracked_clip(
                    clip_idx=clip_idx,
                    str_idx=str_idx,
                    end_idx=end_idx,
                    task_prompt=task_prompt,
                    word=word,
                    masks=masks,
                    scores=scores,
                    logits=logits,
                    boxes=boxes,
                    keypoints=all_keypoints,
                    tracked_keypoints=tracked_keypoints,
                    tracked_visibility=tracked_visibility,
                    video_frames=video_frames,
                    point_to_mask=point_to_mask,
                    global_frames_array=global_frames_array,
                    data_name=data_name,
                    keypoint_types=keypoint_types,
                )
            
if __name__ == "__main__":
    
    from src.datasets.agibotworld import AgiBotWorldDataset
    ds = AgiBotWorldDataset("dataset/AgiBotWorld-Beta")
    
    # get first video
    data_dict = ds[0]

    # load config
    config = load_config("visual_trace")
    pipeline = VisualTracePipeline(config, verbose=True)

    # run pipeline
    results = pipeline.process(data_dict)

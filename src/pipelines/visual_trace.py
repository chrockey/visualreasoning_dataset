from typing import Any, Dict

import numpy as np
from .base import BasePipeline, load_config
from ..models.gemma import Gemma
from ..models.grounded_sam import GroundedSAM2
from ..models.cotracker import CoTracker, KeypointFilter
from ..visualizers.visual_trace import VisualTraceVisualizer

# TODO : Implement the VisualTracePipeline
# 1. Load the Gemma, CoTracker v3, Grounded-SAM2 models
# 2. extract main object from the image using Gemma
# 3. Grounded SAM2 to segment the main object and extract the keypoints
# 4. track the keypoints using CoTracker v3
# 5. filter the keypoints using the confidence score


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
        
        self.keypoint_tracker = CoTracker(config["cotracker"]["model_id"])
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

    def process(self, data_dict: Dict[str, Any]):
        descriptions = data_dict["descriptions"]
        video_frames = data_dict["frames"]
        global_frames_array = None

        gripper_word = self.robot_gripper_prompt + "." if self.track_robot_gripper else None

        for clip_idx, description in enumerate(descriptions):
            str_idx, end_idx, task_prompt = description
            word = self.object_extractor(task_prompt) + "."
            image = video_frames[str_idx]

            if self.verbose:
                print(f"Clip {clip_idx}: Task prompt: '{task_prompt}'")
                print(f"Clip {clip_idx}: Gemma extracted key object: '{word}'")

            # Segment task object
            masks, scores, logits, boxes, labels = self.grounded_segmenter(image, word)
            masks, scores, logits, boxes, labels = self._select_top_masks(
                masks, scores, logits, boxes, labels
            )
            scores = np.asarray(scores).reshape(-1) if scores is not None else None
            logits = np.asarray(logits)
            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = np.squeeze(logits, axis=1)
            self.grounded_segmenter.reset_predictor()
            
            # if no masks found, print error and stop execution
            if masks is None or masks.shape[0] == 0:
                error_msg = (
                    f"[VisualTracePipeline] Grounded SAM2 failed to produce masks "
                    f"for clip {clip_idx} with prompt '{word}'."
                )
                if self.verbose:
                    print(error_msg)
                raise RuntimeError(error_msg)

            # Extract keypoints from task object masks
            task_keypoints = CoTracker.extract_keypoints_from_masks(masks, self.keypoints_per_mask)
            points_per_mask = task_keypoints.shape[1]

            # Track task object keypoints for this clip
            task_tracked_keypoints, task_tracked_visibility = self.keypoint_tracker(
                video_frames[str_idx:end_idx], task_keypoints.reshape(1, -1, 2)
            )

            # Filter task keypoints
            filtered_task_keypoints, filtered_task_visibility, n_task_keypoints = self.keypoint_filter(
                task_tracked_keypoints,
                task_tracked_visibility,
            )

            # Track robot gripper per clip if enabled
            gripper_keypoints = None
            if self.track_robot_gripper:
                gripper_masks, gripper_scores, gripper_logits, gripper_boxes, gripper_labels = self.grounded_segmenter(
                    video_frames[str_idx], gripper_word
                )
                gripper_masks, gripper_scores, gripper_logits, gripper_boxes, gripper_labels = self._select_top_masks(
                    gripper_masks,
                    gripper_scores,
                    gripper_logits,
                    gripper_boxes,
                    gripper_labels,
                )
                self.grounded_segmenter.reset_predictor()

                if gripper_masks is not None and gripper_masks.shape[0] > 0:
                    gripper_scores = np.asarray(gripper_scores).reshape(-1)
                    gripper_logits = np.asarray(gripper_logits)
                    if gripper_logits.ndim == 4 and gripper_logits.shape[1] == 1:
                        gripper_logits = np.squeeze(gripper_logits, axis=1)
                    gripper_keypoints = CoTracker.extract_keypoints_from_masks(gripper_masks, self.keypoints_per_mask)
                    gripper_points_per_mask = gripper_keypoints.shape[1]
                    gripper_tracked_keypoints, gripper_tracked_visibility = self.keypoint_tracker(
                        video_frames[str_idx:end_idx], gripper_keypoints.reshape(1, -1, 2)
                    )
                    if self.verbose:
                        print(
                            f"Clip {clip_idx}: Robot gripper detected with {gripper_keypoints.shape[0]} masks, "
                            "tracking across clip frames"
                        )
                else:
                    if self.verbose:
                        print(f"Clip {clip_idx}: Warning: Robot gripper not detected for this clip")

            # Combine with gripper if available
            keypoint_types = None
            all_keypoints = task_keypoints
            point_to_mask = np.repeat(
                np.arange(task_keypoints.shape[0])[:, None], points_per_mask, axis=1
            ).reshape(-1)

            if self.track_robot_gripper and gripper_keypoints is not None:
                # Filter gripper keypoints
                filtered_gripper_keypoints, filtered_gripper_visibility, n_gripper_keypoints = self.keypoint_filter(
                    gripper_tracked_keypoints,
                    gripper_tracked_visibility,
                )

                # Combine filtered task and gripper keypoints
                filtered_task_keypoints = np.concatenate([filtered_task_keypoints, filtered_gripper_keypoints], axis=2)
                filtered_task_visibility = np.concatenate([filtered_task_visibility, filtered_gripper_visibility], axis=2)

                # Create keypoint_types array
                keypoint_types = np.concatenate([
                    np.zeros(n_task_keypoints, dtype=np.int32),
                    np.ones(n_gripper_keypoints, dtype=np.int32)
                ])

                # Combine for visualization
                all_keypoints = np.concatenate([task_keypoints, gripper_keypoints], axis=0)
                num_task_masks = task_keypoints.shape[0]
                gripper_point_to_mask = np.repeat(
                    np.arange(num_task_masks, num_task_masks + gripper_keypoints.shape[0])[:, None],
                    points_per_mask,
                    axis=1
                ).reshape(-1)
                point_to_mask = np.concatenate([point_to_mask, gripper_point_to_mask])

                # Combine masks
                masks = np.concatenate([masks, gripper_masks], axis=0)
                scores = np.concatenate([scores, gripper_scores], axis=0)
                logits = np.concatenate([logits, gripper_logits], axis=0)
                boxes = np.concatenate([boxes, gripper_boxes], axis=0)
                labels = np.concatenate([labels, gripper_labels], axis=0)

            tracked_keypoints = filtered_task_keypoints
            tracked_visibility = filtered_task_visibility

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

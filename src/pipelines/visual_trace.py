from typing import Any, Dict

import numpy as np
from .base import BasePipeline, load_config
from ..models.gemma import Gemma
from ..models.grounded_sam import GroundedSAM2
from ..models.cotracker import CoTracker
from ..visualizers.visual_trace import VisualTraceVisualizer

# TODO : Implement the VisualTracePipeline
# 1. Load the Gemma, CoTracker v3, Grounded-SAM2 models
# 2. extract main object from the image using Gemma
# 3. Grounded SAM2 to segment the main object and extract the keypoints
# 4. track the keypoints using CoTracker v3
# 5. filter the keypoints using the confidence score


class KeypointFilter:
    # Rule-based filtering system for tracked keypoints

    def __init__(self):
        self.rules = []

    def add_rule(self, rule_fn, **kwargs):

        self.rules.append((rule_fn, kwargs))
        return self

    def apply(self, tracked_keypoints: np.ndarray, tracked_visibility: np.ndarray):
        current_keypoints = tracked_keypoints
        current_visibility = tracked_visibility

        for rule_fn, kwargs in self.rules:
            current_keypoints, current_visibility = rule_fn(
                current_keypoints, current_visibility, **kwargs
            )

        return current_keypoints, current_visibility

    @staticmethod
    def top_moving_keypoints(tracked_keypoints: np.ndarray, tracked_visibility: np.ndarray, top_k: int = 3):
        B, T, N, _ = tracked_keypoints.shape

        if N <= top_k:
            print(f"Number of keypoints ({N}) is less than or equal to top_k ({top_k}). No filtering applied.")
            return tracked_keypoints, tracked_visibility, np.arange(N)

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

        return filtered_keypoints, filtered_visibility


class VisualTracePipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any], verbose: bool = True, suffix: str = "base"):
        super().__init__(config)
        self.object_extractor = Gemma(config["gemma"]["model_id"])
        
        # Initialize GroundedSam with config
        grounded_sam_config = config.get("grounded_sam", {})
        self.grounded_segmenter = GroundedSAM2(**grounded_sam_config)
        
        self.keypoint_tracker = CoTracker(config["cotracker"]["model_id"])
        self.verbose = verbose
        self.visualizer = VisualTraceVisualizer(
            save_dir=f"viz/visual_trace_{suffix}",
            tracks_leave_trace=-1,
            save_per_mask=False,  # Disable per-mask video saving (all keypoints video always enabled)
        )

        # Initialize keypoint filter
        self.keypoint_filter = KeypointFilter()
        self.keypoint_filter.add_rule(KeypointFilter.top_moving_keypoints, top_k=3)

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        descriptions = data_dict["descriptions"]
        video_frames = data_dict["frames"]
        global_frames_array = None

        for clip_idx, description in enumerate(descriptions):
            str_idx, end_idx, task_prompt = description
            word = self.object_extractor(task_prompt) + "."
            image = video_frames[str_idx]
            
            # 
            masks, scores, logits, boxes, labels = self.grounded_segmenter(image, word)
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
            
            # Extract keypoints from masks
            keypoints = CoTracker.extract_keypoints_from_masks(masks)  # (n, 3, 2)
            num_masks, points_per_mask = keypoints.shape[:2]
            point_to_mask = np.repeat(
                np.arange(num_masks)[:, None], points_per_mask, axis=1
            ).reshape(-1)

            # TODO: Pass keypoints to keypoint_tracker
            tracked_keypoints, tracked_visibility = self.keypoint_tracker(
                video_frames[str_idx:end_idx], keypoints.reshape(1, -1, 2)
            )
            
            # TODO: Apply rule-based filtering of tracked_keypoints
            tracked_keypoints, tracked_visibility = self.keypoint_filter.apply(
                tracked_keypoints, tracked_visibility
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
                    keypoints=keypoints,
                    tracked_keypoints=tracked_keypoints,
                    tracked_visibility=tracked_visibility,
                    video_frames=video_frames,
                    point_to_mask=point_to_mask,
                    global_frames_array=global_frames_array,
                    data_name=data_name,
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

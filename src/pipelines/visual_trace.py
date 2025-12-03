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
        self.keypoint_filter = KeypointFilter(
            traj_top_k=config["keypoint_filter"]["traj_top_k"],
            drop_length_ratio_threshold=config["keypoint_filter"]["traj_drop_len_ratio"],
            drop_use_median=config["keypoint_filter"]["traj_drop_use_median"],
        )

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

            # Extract keypoints from task object masks
            task_keypoints = CoTracker.extract_keypoints_from_masks(masks, self.keypoints_per_mask)
            points_per_mask = task_keypoints.shape[1]

            # Track task object keypoints for this clip
            task_tracked_keypoints, task_tracked_visibility = self.keypoint_tracker(
                video_frames[str_idx:end_idx], task_keypoints.reshape(1, -1, 2)
            )
            
            # TODO: Apply rule-based filtering of tracked_keypoints
            tracked_keypoints, tracked_visibility, n_keypoints = self.keypoint_filter(tracked_keypoints, tracked_visibility)

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

from pathlib import Path
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
        )
        
    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        descriptions = data_dict["description"]
        frame_sets = data_dict["frames"]

        if len(descriptions) != len(frame_sets):
            error_msg = (
                "[VisualTracePipeline] Mismatch between number of descriptions "
                f"({len(descriptions)}) and frame segments ({len(frame_sets)})."
            )
            if self.verbose:
                print(error_msg)
            raise RuntimeError(error_msg)

        clip_start_frame = 0
        
        for clip_idx, (description, frame_set) in enumerate(zip(descriptions, frame_sets)):
            clip_frames = frame_set
            if isinstance(clip_frames, (list, tuple)):
                if len(clip_frames) == 0:
                    continue
                clip_frames = np.stack(list(clip_frames), axis=0)

            if len(clip_frames) == 0:
                continue
            
            clip_len = clip_frames.shape[0]
            start_frame = clip_start_frame
            end_frame = start_frame + clip_len
            clip_start_frame = end_frame

            word = self.object_extractor(description) + "."
            image = clip_frames[0]
            
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
                clip_frames, keypoints.reshape(1, -1, 2)
            )
            
            if self.verbose:
                clip_dir = Path(self.visualizer.save_dir) / f"clip_{clip_idx:04d}"
                clip_dir.mkdir(parents=True, exist_ok=True)
                info_path = clip_dir / "clip_info.txt"
                info_content = [
                    f"clip_index: {clip_idx}",
                    f"start_frame: {start_frame}",
                    f"end_frame: {end_frame}",
                    f"description: {description}",
                ]
                info_path.write_text("\n".join(info_content), encoding="utf-8")

                print(description)
                print(word)
                print(masks.shape, scores.shape, logits.shape, boxes.shape)
                print(f"Keypoints shape: {keypoints.shape}")
                print(f"Keypoints for first mask: {keypoints[0]}")
                print(
                    f"Total tracked points: {tracked_keypoints.shape[2]} across {num_masks} masks"
                )

                track_sequence = tracked_keypoints[0]
                for mask_idx in range(num_masks):
                    mask_point_indices = np.where(point_to_mask == mask_idx)[0]
                    if mask_point_indices.size == 0:
                        continue
                    mask_point_list = mask_point_indices.tolist()

                    video_name = f"clip_{clip_idx:04d}_mask_{mask_idx:02d}"
                    mask_array = masks[mask_idx]
                    for frame_offset, frame in enumerate(clip_frames):
                        keypoints_frame = track_sequence[frame_offset, mask_point_list]
                        self.visualizer.save_visualizations(
                            frame,
                            mask_array,
                            keypoints_frame,
                            video_name=video_name,
                            frame_idx=start_frame + frame_offset,
                            finalize=frame_offset == len(clip_frames) - 1,
                        )
                    trace_video_name = f"clip_{clip_idx:04d}_mask_{mask_idx:02d}_trace"
                    self.visualizer.save_trace_video(
                        clip_frames,
                        track_sequence[:, mask_point_list],
                        video_name=trace_video_name,
                        mask=mask_array,
                    )
                print(tracked_keypoints.shape, tracked_visibility.shape)
            
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

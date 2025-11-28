from typing import Any, Dict, List, Tuple
import os

import numpy as np
from PIL import Image, ImageDraw
import cv2

from src.models.gemma import Gemma
from src.models.grounded_sam2 import GroundedSAM2
from src.models.molmo import Molmo
from src.models.sam2 import SAM2
from src.utils.affordance_type1_utils import mask_to_bbox, find_best_interacting_object, sample_interaction_points, create_demo_video, visualize_affordance, visualize_video_frame

from .base import BasePipeline, load_config
from tqdm import tqdm


class AffordanceType1Pipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mode = config.get('mode', 'image')
        self.gemma = Gemma(**config.get("gemma", {}))
        self.grounded_sam2 = GroundedSAM2(**config.get("grounded_sam2", {}))
        self.sam2 = SAM2(**config.get("sam2", {}), mode =  self.mode)

    def preprocess(self, data_dict: Dict[str, Any]):
        """Prepare input data for processing."""
        return data_dict
    
    def _process_image_mode(self, frames, description, main_object, text_prompt, vis_dir, data_dict):
        """Process frames independently in image mode."""
        results = []

        # Process each frame
        print(f"Processing {len(frames)} frames with Grounded SAM2...")
        for i, frame in enumerate(frames):
            frame_result = {
                "frame_idx": i,
                "main_object": main_object,
                "masks": None,
                "bboxes": None,
                "labels": None,
                "visualization_path": None
            }

            # Step 2 : Detect object and hand/gripper using Grounded SAM2
            masks, scores, logits, boxes, labels = self.grounded_sam2(frame, text_prompt)
            if len(boxes) > 0:
                # Step 3 : Find sample interaction points from best object
                best_object_idx, object_indices, hand_gripper_indices, max_overlap = find_best_interacting_object(
                    masks, boxes, labels
                )
                if best_object_idx is not None and max_overlap > 0:
                    hand_gripper_masks = [masks[idx] for idx in hand_gripper_indices]
                    interaction_points = sample_interaction_points(
                        masks[best_object_idx],
                        hand_gripper_masks,
                        num_points=5
                    )
                    print(f"Frame {i}: Found {len(interaction_points)} interaction points on {labels[best_object_idx]} (overlap: {max_overlap:.0f} pixels)")

                frame_result.update({
                    "masks": masks.tolist(),
                    "scores": scores.tolist(),
                    "bboxes": boxes.tolist(),  # boxes are in xyxy format
                    "labels": labels,
                    "interaction_points": interaction_points.tolist() if interaction_points is not None else None,
                    "best_object_idx": best_object_idx,
                    "overlap_area": max_overlap
                })

                print(f"Frame {i}: Detected {len(boxes)} objects - {labels}")
            else:
                print(f"Frame {i}: No objects detected")

            # Visualize and save
            if len(boxes) > 0:
                vis_img = visualize_affordance(frame, boxes, masks, labels, interaction_points)
            else:
                vis_img = Image.fromarray(frame.astype(np.uint8))

            vis_path = os.path.join(vis_dir, f"frame_{i:04d}.png")
            vis_img.save(vis_path)
            frame_result["visualization_path"] = vis_path

            results.append(frame_result)

            # Reset predictor for next frame
            self.grounded_sam2.reset_predictor()

        return {
            "video_name": data_dict["video_name"],
            "description": description,
            "main_object": main_object,
            "metadata": data_dict.get("metadata", {}),
            "results": results
        }

    def _process_video_mode(self, frames, description, main_object, text_prompt, vis_dir, data_dict):
        """
        Process video with interaction detection and bidirectional propagation.

        Steps:
        1. Scan frames to find first interaction (overlap between object and hand/gripper)
        2. Sample interaction points from that frame
        3. Use SAM2 video tracking to propagate segmentation bidirectionally
        """
        print(f"VIDEO MODE: Scanning {len(frames)} frames for first interaction...")

        # Step 1: Find first interaction frame
        first_interaction_frame = None
        interaction_points = None
        best_object_mask = None

        for i, frame in enumerate(frames):
            # Step 2 :  Detect object and hand/gripper using Grounded SAM2
            masks, scores, logits, boxes, labels = self.grounded_sam2(frame, text_prompt)

            if len(boxes) == 0:
                self.grounded_sam2.reset_predictor()
                continue

            # Step 3 : Find first interaction frame and sample interaction points from best object
            best_object_idx, object_indices, hand_gripper_indices, max_overlap = find_best_interacting_object(
                masks, boxes, labels
            )
            if best_object_idx is not None and max_overlap > 50:  # Threshold for interaction
                hand_gripper_masks = [masks[idx] for idx in hand_gripper_indices]
                first_interaction_frame = i
                interaction_points = sample_interaction_points(
                    masks[best_object_idx],
                    hand_gripper_masks,
                    num_points=5
                )
                print(f"Found first interaction at frame {i} with {len(interaction_points)} points (overlap: {max_overlap:.0f} pixels)")
                print(f"Interaction object: {labels[best_object_idx]}")
                break

            self.grounded_sam2.reset_predictor()

        # Step 4 : Use SAM2 video tracking to propagate part-level segmentation.
        print(f"Propagating segmentation from frame {first_interaction_frame} across entire video...")
        video_segments = self.sam2(frames, interaction_points, first_interaction_frame)

        results = []
        for frame_idx in range(len(frames)):
            frame_result = {
                "frame_idx": frame_idx,
                "main_object": main_object,
                "masks": None,
                "bboxes": None,
                "interaction_points": interaction_points.tolist() if frame_idx == first_interaction_frame else None,
                "is_interaction_frame": frame_idx == first_interaction_frame,
                "visualization_path": None
            }

            propagated_mask = video_segments.get(frame_idx)
            bbox = None
            if propagated_mask is not None:
                if propagated_mask.ndim == 2:
                    propagated_mask = propagated_mask[np.newaxis, ...]  # Add batch dimension

                # Compute bounding box from mask
                mask_for_bbox = propagated_mask[0] if propagated_mask.ndim == 3 else propagated_mask
                bbox = mask_to_bbox(mask_for_bbox)

                frame_result.update({
                    "masks": propagated_mask.tolist(),
                    "bboxes": bbox
                })

            # Visualize
            mask_for_vis = propagated_mask[0] if propagated_mask is not None and propagated_mask.ndim == 3 else propagated_mask
            points_to_draw = interaction_points if frame_idx == first_interaction_frame else None
            vis_img = visualize_video_frame(frames[frame_idx], mask_for_vis, bbox, points_to_draw)

            vis_path = os.path.join(vis_dir, f"frame_{frame_idx:04d}.png")
            vis_img.save(vis_path)
            frame_result["visualization_path"] = vis_path

            results.append(frame_result)

        # Create demo video
        video_output_path = os.path.join(vis_dir, "demo_video.mp4")
        create_demo_video(vis_dir, video_output_path, fps=10, first_interaction_frame=first_interaction_frame)

        return {
            "video_name": data_dict["video_name"],
            "description": description,
            "main_object": main_object,
            "first_interaction_frame": first_interaction_frame,
            "metadata": data_dict.get("metadata", {}),
            "results": results
        }
        
    def process(self, data_dict: Dict[str, Any]):
        """
        Main pipeline for grasp affordance extraction:
        1. Extract main manipulated object from description using Gemma
        2. Detect object and hand/gripper using Grounded SAM2
        3. Find first interaction frame and sample interaction points
        4. Use SAM2 video tracking to propagate part-level segmentation bidirectionally
        """
        frames = data_dict["frames"]  # (N, H, W, 3)
        description = data_dict['description']

        # Step 1: Extract main manipulated object using Gemma
        print(f"Extracting main object from description: {description}")
        main_object = self.gemma(description).strip()
        print(f"Detected main object: {main_object}")
        text_prompt = f"{main_object}. hand. gripper."
        
        # Create save directory for visualizations
        save_dir = self.config.get("save_dir", ".")
        vis_dir = os.path.join(save_dir, "visualizations", data_dict["video_name"])
        os.makedirs(vis_dir, exist_ok=True)

        results = []
        if self.mode == "image":
            # IMAGE MODE: Process each frame independently
            return self._process_image_mode(frames, description, main_object, text_prompt, vis_dir, data_dict)
        elif self.mode == "video":
            # VIDEO MODE: Find interaction frame and propagate
            return self._process_video_mode(frames, description, main_object, text_prompt, vis_dir, data_dict)


if __name__ == "__main__":
    import argparse
    from src.datasets.egodex import EgoDexDataset
    from src.datasets.agibotworld import AgiBotWorldDataset

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test AffordanceType1 pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["egodex", "agibotworld"],
        default="egodex",
        help="Dataset to use for testing (default: egodex)"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of video to process (default: 0)"
    )
    parser.add_argument(
        "--action-index",
        type=int,
        default=0,
        help="For AgiBotWorld: index of action to process (default: 0)"
    )
    args = parser.parse_args()

    # Load pipeline configuration and create pipeline
    config = load_config("affordance_type1")
    pipeline = AffordanceType1Pipeline(config)

    # Load dataset based on argument
    if args.dataset == "egodex":
        print("Loading EgoDex dataset...")
        dataset = EgoDexDataset()
        if len(dataset) == 0:
            print("No EgoDex data found. Please check data directory.")
            exit(1)

        # Get video sample
        data_dict = dataset[args.index]
        print(f"Testing pipeline with video: {data_dict['video_name']}")
        print(f"Description: {data_dict['description']}")
        print(f"Frames shape: {data_dict['frames'].shape}")

        # Run pipeline
        print("\nRunning affordance type2 pipeline...")
        results = pipeline(data_dict, save_dir=".")

    elif args.dataset == "agibotworld":
        print("Loading AgiBotWorld dataset...")
        dataset = AgiBotWorldDataset()
        if len(dataset) == 0:
            print("No AgiBotWorld data found. Please check data directory.")
            exit(1)

        # Get video sample (contains multiple actions)
        sample = dataset[args.index]
        print(f"Testing pipeline with video: {sample['video_name']}")
        print(f"Total actions in video: {len(sample['frames'])}")

        # Select specific action
        action_idx = args.action_index
        if action_idx >= len(sample['frames']):
            print(f"Action index {action_idx} out of range. Using action 0.")
            action_idx = 0

        # Prepare data dict for pipeline (AgiBotWorld has per-action frames)
        data_dict = {
            "video_name": f"{sample['video_name']}_action{action_idx}",
            "frames": sample['frames'][action_idx],  # Get frames for specific action
            "description": sample['description'][action_idx],  # Get description for specific action
            "metadata": sample['metadata']
        }

        print(f"\nProcessing action {action_idx}:")
        print(f"Description: {data_dict['description']}")
        print(f"Frames shape: {data_dict['frames'].shape}")

        # Run pipeline
        print("\nRunning affordance type2 pipeline...")
        results = pipeline(data_dict, save_dir=".")

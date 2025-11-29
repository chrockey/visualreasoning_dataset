from typing import Any, Dict, List, Tuple
import os
import gc

import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch

from src.models.gemma import Gemma
from src.models.grounded_sam2 import GroundedSAM2
from src.models.molmo import Molmo
from src.models.sam2 import SAM2
from src.utils.affordance_type1_utils import mask_to_bbox, find_best_interacting_object, sample_interaction_points, create_demo_video, visualize_affordance, visualize_video_frame

from .base import BasePipeline, load_config, load_dataset_from_config
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

            # Clean up memory after processing each frame
            del masks, scores, logits, boxes, labels, vis_img
            torch.cuda.empty_cache()
            gc.collect()

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
                # Clean up memory
                del masks, scores, logits, boxes, labels
                torch.cuda.empty_cache()
                gc.collect()
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
            # Clean up memory after processing each frame
            del masks, scores, logits, boxes, labels
            torch.cuda.empty_cache()
            gc.collect()

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

        # Create demo video with caption
        video_output_path = os.path.join(vis_dir, "demo_video.mp4")
        create_demo_video(vis_dir, video_output_path, fps=10,
                         first_interaction_frame=first_interaction_frame,
                         caption=description)

        return {
            "video_name": data_dict["video_name"],
            "description": description,
            "main_object": main_object,
            "first_interaction_frame": first_interaction_frame,
            "metadata": data_dict.get("metadata", {}),
            "results": results
        }
        
    def _process_segment(
        self,
        frames: np.ndarray,
        segment_idx: int,
        start_frame: int,
        end_frame: int,
        description: str,
        video_name: str,
        metadata: Dict[str, Any],
        dataset_name: str = None
    ) -> Dict[str, Any]:
        """Process a single temporal segment.

        Args:
            frames: Full video frames (N, H, W, 3)
            segment_idx: Index of this segment
            start_frame: Start frame (inclusive)
            end_frame: End frame (inclusive)
            description: Text description for this segment
            video_name: Name of the video
            metadata: Video metadata
            dataset_name: Name of the dataset (optional, for organizing output)

        Returns:
            Results dict for this segment
        """
        # Slice frames to the segment of interest (end_frame is INCLUSIVE)
        segment_frames = frames[start_frame:end_frame+1]

        print(f"\nProcessing segment {segment_idx}: [{start_frame}:{end_frame}] (inclusive) with {len(segment_frames)} frames")
        print(f"Description: {description}")

        # Step 1: Extract main manipulated object using Gemma
        print(f"Extracting main object from description: {description}")
        main_object = self.gemma(description).strip()
        print(f"Detected main object: {main_object}")
        text_prompt = f"{main_object}. hand. gripper."

        # Create save directory for visualizations
        save_dir = self.config.get("save_dir", ".")

        # Sanitize video_name: replace "/" with "_" to avoid deep directory nesting
        sanitized_video_name = video_name.replace("/", "_")

        if dataset_name:
            vis_dir = os.path.join(save_dir, "visualizations", dataset_name, sanitized_video_name, f"segment_{segment_idx}")
        else:
            vis_dir = os.path.join(save_dir, "visualizations", sanitized_video_name, f"segment_{segment_idx}")
        os.makedirs(vis_dir, exist_ok=True)

        # Save description text file
        description_path = os.path.join(vis_dir, "description.txt")
        with open(description_path, 'w') as f:
            f.write(f"Description: {description}\n")
            f.write(f"Main Object: {main_object}\n")
            f.write(f"Segment: {segment_idx}\n")
            f.write(f"Frame Range: [{start_frame}:{end_frame}] (inclusive)\n")

        # Build data_dict for this segment
        segment_data_dict = {
            "video_name": f"{video_name}/segment_{segment_idx}",
            "metadata": metadata
        }

        # Process based on mode
        if self.mode == "image":
            # IMAGE MODE: Process each frame independently
            segment_result = self._process_image_mode(
                segment_frames, description, main_object, text_prompt, vis_dir, segment_data_dict
            )
        elif self.mode == "video":
            # VIDEO MODE: Find interaction frame and propagate
            segment_result = self._process_video_mode(
                segment_frames, description, main_object, text_prompt, vis_dir, segment_data_dict
            )

        # Add segment metadata
        segment_result.update({
            "segment_idx": segment_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

        return segment_result

    def process(self, data_dict: Dict[str, Any]):
        """
        Main pipeline for grasp affordance extraction:
        1. Extract main manipulated object from description using Gemma
        2. Detect object and hand/gripper using Grounded SAM2
        3. Find first interaction frame and sample interaction points
        4. Use SAM2 video tracking to propagate part-level segmentation bidirectionally

        Data format:
        - data_dict['frames']: (N, H, W, 3) whole video
        - data_dict['descriptions']: List[(start_frame_inclusive, end_frame_inclusive, description_text)]
        """
        frames = data_dict["frames"]  # (N, H, W, 3)
        descriptions = data_dict['descriptions']  # List[(start, end, text)]
        video_name = data_dict["video_name"]
        metadata = data_dict.get("metadata", {})

        # Get dataset name from config
        dataset_name = self.config.get("dataset", {}).get("name", None)

        print(f"Processing video: {video_name}")
        print(f"Total segments: {len(descriptions)}")

        # Process all segments
        all_segments = []
        for segment_idx, (start_frame, end_frame, description) in enumerate(descriptions):
            segment_result = self._process_segment(
                frames, segment_idx, start_frame, end_frame,
                description, video_name, metadata, dataset_name
            )
            all_segments.append(segment_result)

        # Return aggregated results
        return {
            "video_name": video_name,
            "metadata": metadata,
            "num_segments": len(all_segments),
            "segments": all_segments
        }


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test AffordanceType1 pipeline")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of video to process (default: 0)"
    )
    parser.add_argument(
        "-s", "--segment-index",
        type=int,
        default=None,
        help="Process only a specific segment index. If not provided, processes all segments (default: None)"
    )
    parser.add_argument(
        "-d", "--dataset-name",
        type=str,
        default=None,
        help="Override dataset name from config (e.g., egodex, oxe, agibotworld, holoassist)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory from config"
    )
    args = parser.parse_args()

    # Load pipeline configuration and create pipeline
    config = load_config("affordance_type1")

    # Override dataset config if arguments provided
    if args.dataset_name is not None:
        if "dataset" not in config:
            config["dataset"] = {}
        config["dataset"]["name"] = args.dataset_name

    if args.dataset_dir is not None:
        if "dataset" not in config:
            config["dataset"] = {}
        config["dataset"]["dir"] = args.dataset_dir

    pipeline = AffordanceType1Pipeline(config)

    # Load dataset from config
    print("Loading dataset from config...")
    dataset = load_dataset_from_config(config)

    # Get video sample
    data_dict = dataset[args.index]
    print(f"Testing pipeline with video: {data_dict['video_name']}")
    print(f"Frames shape: {data_dict['frames'].shape}")
    print(f"Total segments: {len(data_dict['descriptions'])}")
    print(f"Descriptions: {data_dict['descriptions']}")

    # Filter to specific segment if requested
    if args.segment_index is not None:
        if args.segment_index >= len(data_dict['descriptions']):
            print(f"Error: Segment index {args.segment_index} out of range (0-{len(data_dict['descriptions'])-1})")
            exit(1)

        print(f"\nProcessing only segment {args.segment_index}")
        # Keep only the selected segment
        data_dict['descriptions'] = [data_dict['descriptions'][args.segment_index]]

    # Run pipeline
    print("\nRunning affordance type1 pipeline...")
    results = pipeline(data_dict, save_dir=".")

    # TODO: Resolve Out-of-memory error when processing too long videos

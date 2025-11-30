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
                # Step 3 : Find sample interaction points from best objects per hand
                best_objects, object_indices, hand_gripper_indices, overlaps = find_best_interacting_object(
                    masks, boxes, labels
                )
                
                # Sample interaction points for all hands that have interactions
                all_interaction_points = []
                interaction_details = []
                
                for hand_type in ['left', 'right']:
                    best_obj_idx = best_objects[hand_type]
                    overlap = overlaps[hand_type]
                    hand_idx = hand_gripper_indices[hand_type]
                    
                    if best_obj_idx is not None and overlap > 0 and hand_idx is not None:
                        hand_masks = [masks[hand_idx]]
                        interaction_points = sample_interaction_points(
                            masks[best_obj_idx],
                            hand_masks,
                            num_points=5
                        )
                        
                        all_interaction_points.extend(interaction_points)
                        interaction_details.append({
                            'hand_type': hand_type,
                            'object_idx': best_obj_idx,
                            'object_label': labels[best_obj_idx],
                            'overlap': overlap,
                            'points': interaction_points
                        })
                        
                        print(f"Frame {i}: Found {len(interaction_points)} {hand_type} hand interaction points on {labels[best_obj_idx]} (overlap: {overlap:.0f} pixels)")
                
                # Convert to numpy array for visualization
                all_interaction_points = np.array(all_interaction_points) if all_interaction_points else None

                frame_result.update({
                    "masks": masks.tolist(),
                    "scores": scores.tolist(),
                    "bboxes": boxes.tolist(),  # boxes are in xyxy format
                    "labels": labels,
                    "interaction_points": all_interaction_points.tolist() if all_interaction_points is not None else None,
                    "interaction_details": interaction_details,
                    "best_objects": best_objects,
                    "overlaps": overlaps,
                    "hand_gripper_indices": hand_gripper_indices
                })

                print(f"Frame {i}: Detected {len(boxes)} objects - {labels}")
            else:
                print(f"Frame {i}: No objects detected")

            # Visualize and save
            if len(boxes) > 0:
                vis_img = visualize_affordance(frame, boxes, masks, labels, all_interaction_points)
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
        """
        print(f"VIDEO MODE: Scanning {len(frames)} frames for all interactions...")

        # Step 1: Find all interaction frames for each hand
        hand_interactions = {'left': [], 'right': []}  # List of (frame_idx, interaction_points, details)
        
        # Create separate directory for masks
        affordance_dir = os.path.join(vis_dir, "affordance")
        os.makedirs(affordance_dir, exist_ok=True)
        
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

            # Visualize affordance for each frame
            # vis_affordance_img = visualize_affordance(frame, boxes, masks, labels)
            # affordance_vis_path = os.path.join(affordance_dir, f"affordance_frame_{i:04d}.png")
            # vis_affordance_img.save(affordance_vis_path)

            # Step 3 : Find interaction frames and sample interaction points for each hand separately
            best_objects, object_indices, hand_gripper_indices, overlaps = find_best_interacting_object(
                masks, boxes, labels
            )
            
            frame_interactions = []
            
            for hand_type in ['left', 'right']:
                best_obj_idx = best_objects[hand_type]
                overlap = overlaps[hand_type]
                hand_idx = hand_gripper_indices[hand_type]
                
                if best_obj_idx is not None and overlap > 50 and hand_idx is not None:  # Threshold for interaction
                    hand_masks = [masks[hand_idx]]
                    interaction_points = sample_interaction_points(
                        masks[best_obj_idx],
                        hand_masks,
                        num_points=5
                    )
                    
                    interaction_detail = {
                        'frame_idx': i,
                        'hand_type': hand_type,
                        'object_idx': best_obj_idx,
                        'object_label': labels[best_obj_idx],
                        'overlap': overlap,
                        'points': interaction_points,
                        'hand_idx': hand_idx,
                        'object_mask': masks[best_obj_idx]  # Store the original object mask
                    }
                    
                    hand_interactions[hand_type].append(interaction_detail)
                    frame_interactions.append(interaction_detail)
                    
                    print(f"Found {hand_type} hand interaction at frame {i}: {len(interaction_points)} points on {labels[best_obj_idx]} (overlap: {overlap:.0f} pixels)")
            
            # Save affordance visualization with interaction points if any found
            if frame_interactions:
                all_points = []
                for interaction in frame_interactions:
                    all_points.extend(interaction['points'])
                
                all_points = np.array(all_points) if all_points else np.array([])
                # vis_affordance_with_points = visualize_affordance(frame, boxes, masks, labels, all_points)
                # affordance_interaction_path = os.path.join(affordance_dir, f"affordance_interaction_frame_{i:04d}.png")
                # vis_affordance_with_points.save(affordance_interaction_path)
            
            self.grounded_sam2.reset_predictor()
            # Clean up memory after processing each frame
            del masks, scores, logits, boxes, labels
            torch.cuda.empty_cache()
            gc.collect()
            
        print(f"Left hand interactions found at frames: {[x['frame_idx'] for x in hand_interactions['left']]}")
        print(f"Right hand interactions found at frames: {[x['frame_idx'] for x in hand_interactions['right']]}")
        
        # Step 4: Perform backward propagation for each hand separately using SAM2 reverse mode
        hand_segments = {'left': {}, 'right': {}}
        
        for hand_type in ['left', 'right']:
            interactions = hand_interactions[hand_type]
            if not interactions:
                continue
                
            print(f"\nProcessing {hand_type} hand interactions...")
            
            # Process each interaction segment (from current interaction backward to previous interaction)
            for i, current_interaction in enumerate(interactions):
                current_frame = current_interaction['frame_idx']
                interaction_points = current_interaction['points']
                
                # Determine start frame for backward propagation
                if i == 0:
                    # First interaction: propagate backward to frame 0
                    start_frame = 0
                else:
                    # Subsequent interactions: propagate backward to previous interaction frame + 1
                    start_frame = interactions[i-1]['frame_idx'] + 1
                
                if current_frame > start_frame:
                    print(f"  Backward propagating {hand_type} hand from frame {current_frame} to {start_frame}")
                    
                    # Use SAM2 reverse propagation mode
                    # SAM2 with reverse=True will propagate from current_frame backward to start_frame
                    backward_segments = self.sam2(
                        frames, 
                        interaction_points, 
                        current_frame,
                        reverse=True,
                        start_frame=start_frame,
                        end_frame=current_frame
                    )
                    
                    # Store results for this hand
                    for frame_idx, mask in backward_segments.items():
                        if start_frame <= frame_idx <= current_frame:
                            hand_segments[hand_type][frame_idx] = {
                                'mask': mask,
                                'interaction_frame': current_frame,
                                'interaction_detail': current_interaction
                            }
                
                print(f"  Completed backward propagation for {hand_type} hand from frame {current_frame}")
        
        # Combine segments from both hands
        combined_segments = {}
        for frame_idx in range(len(frames)):
            frame_data = {}
            
            # Get left hand data
            if frame_idx in hand_segments['left']:
                frame_data['left_mask'] = hand_segments['left'][frame_idx]['mask']
                frame_data['left_interaction_detail'] = hand_segments['left'][frame_idx]['interaction_detail']
            else:
                frame_data['left_mask'] = None
                frame_data['left_interaction_detail'] = None
                
            # Get right hand data  
            if frame_idx in hand_segments['right']:
                frame_data['right_mask'] = hand_segments['right'][frame_idx]['mask']
                frame_data['right_interaction_detail'] = hand_segments['right'][frame_idx]['interaction_detail']
            else:
                frame_data['right_mask'] = None
                frame_data['right_interaction_detail'] = None
                
            combined_segments[frame_idx] = frame_data

        results = []
        for frame_idx in range(len(frames)):
            frame_data = combined_segments[frame_idx]
            
            # Determine if this is an interaction frame and get interaction points
            is_left_interaction = any(x['frame_idx'] == frame_idx for x in hand_interactions['left'])
            is_right_interaction = any(x['frame_idx'] == frame_idx for x in hand_interactions['right'])
            is_interaction_frame = is_left_interaction or is_right_interaction
            
            interaction_points_to_draw = []
            if is_left_interaction:
                left_interaction = next(x for x in hand_interactions['left'] if x['frame_idx'] == frame_idx)
                interaction_points_to_draw.extend(left_interaction['points'])
            if is_right_interaction:
                right_interaction = next(x for x in hand_interactions['right'] if x['frame_idx'] == frame_idx)
                interaction_points_to_draw.extend(right_interaction['points'])
            
            frame_result = {
                "frame_idx": frame_idx,
                "main_object": main_object,
                "left_mask": frame_data['left_mask'].tolist() if frame_data['left_mask'] is not None else None,
                "right_mask": frame_data['right_mask'].tolist() if frame_data['right_mask'] is not None else None,
                "left_interaction_detail": frame_data['left_interaction_detail'],
                "right_interaction_detail": frame_data['right_interaction_detail'],
                "interaction_points": interaction_points_to_draw if interaction_points_to_draw else None,
                "is_interaction_frame": is_interaction_frame,
                "visualization_path": None
            }

            # For interaction frames, use the original object masks from GroundedSAM2
            combined_mask = None
            combined_bbox = None
            
            if is_interaction_frame:
                # Get the original object masks for this interaction frame
                interaction_object_masks = []
                
                if is_left_interaction:
                    left_interaction = next(x for x in hand_interactions['left'] if x['frame_idx'] == frame_idx)
                    # Re-run GroundedSAM2 to get the object mask for this frame
                    # OR use stored masks if available in interaction_detail
                    if 'object_mask' in left_interaction:
                        interaction_object_masks.append(left_interaction['object_mask'])
                        
                if is_right_interaction:
                    right_interaction = next(x for x in hand_interactions['right'] if x['frame_idx'] == frame_idx)
                    if 'object_mask' in right_interaction:
                        interaction_object_masks.append(right_interaction['object_mask'])
                
                # Combine interaction object masks
                if interaction_object_masks:
                    if len(interaction_object_masks) == 1:
                        combined_mask = interaction_object_masks[0]
                    else:
                        combined_mask = np.logical_or(*interaction_object_masks)
            else:
                # For non-interaction frames, use backward propagated masks
                if frame_data['left_mask'] is not None and frame_data['right_mask'] is not None:
                    # Both hands have masks - combine them
                    combined_mask = np.logical_or(frame_data['left_mask'], frame_data['right_mask'])
                elif frame_data['left_mask'] is not None:
                    combined_mask = frame_data['left_mask']
                elif frame_data['right_mask'] is not None:
                    combined_mask = frame_data['right_mask']
            
            if combined_mask is not None:
                combined_bbox = mask_to_bbox(combined_mask)

            # Visualize with combined mask and interaction points
            points_for_vis = np.array(interaction_points_to_draw) if interaction_points_to_draw else None
            vis_img = visualize_video_frame(frames[frame_idx], combined_mask, combined_bbox, points_for_vis)

            vis_path = os.path.join(vis_dir, f"frame_{frame_idx:04d}.png")
            vis_img.save(vis_path)
            frame_result["visualization_path"] = vis_path

            results.append(frame_result)

        # Create demo video with caption - use first interaction frame found
        all_interaction_frames = [x['frame_idx'] for x in hand_interactions['left']] + [x['frame_idx'] for x in hand_interactions['right']]
        first_interaction_frame = min(all_interaction_frames) if all_interaction_frames else None
        
        video_output_path = os.path.join(vis_dir, "demo_video.mp4")
        create_demo_video(vis_dir, video_output_path, fps=10,
                         caption=description)

        return {
            "video_name": data_dict["video_name"],
            "description": description,
            "main_object": main_object,
            "left_interactions": hand_interactions['left'],
            "right_interactions": hand_interactions['right'],
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

from typing import Any, Dict, List, Tuple
import os
import gc
import json
import copy
import shutil

import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch

from src.models.gemma import Gemma
from src.models.grounded_sam2 import GroundedSAM2
from src.models.grounded_sam2_video_tracker import GroundedSAM2VideoTracker
# from src.models.grounded_sam3_video_tracker import GroundedSAM3VideoTracker
from src.models.molmo import Molmo
from src.models.sam2 import SAM2
from src.utils.affordance_type1_utils import mask_to_bbox, find_best_interacting_object, sample_interaction_points, create_demo_video, visualize_affordance, visualize_video_frame
from src.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from src.utils.common_utils import CommonUtils

from .base import BasePipeline, load_config, load_dataset_from_config
from tqdm import tqdm


class AffordanceType1Pipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mode = config.get('mode', 'image')
        self.gemma = Gemma(**config.get("gemma", {}))
        self.grounded_sam2 = GroundedSAM2(**config.get("grounded_sam2", {}))
        self.sam2 = SAM2(**config.get("sam2", {}), mode =  self.mode)

        # Initialize video tracker for continuous ID tracking mode
        if self.mode == 'video_instance_seg' or self.mode == 'all_objects':
            sam2_config = config.get("sam2", {})
            self.video_tracker = GroundedSAM2VideoTracker(
                grounded_sam2=self.grounded_sam2,
                sam2_checkpoint=config.get("sam2_checkpoint", None),
                model_cfg=config.get("sam2_model_cfg", None),
                model_id=sam2_config.get("model_id", "facebook/sam2-hiera-large"),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

        if self.mode == 'video_instance_seg_sam3':
            sam3_config = config.get("sam3", {})
            self.video_tracker = GroundedSAM3VideoTracker(
                grounded_sam2=self.grounded_sam2,
                gpus_to_use=sam3_config.get("gpus_to_use", None),
                device="cuda" if torch.cuda.is_available() else "cpu",
                model_id=sam3_config.get("model_id", "facebook/sam3")
            )



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

    def _process_video_mode_with_continuous_id(
        self,
        frames: np.ndarray,
        description: str,
        main_object: str,
        text_prompt: str,
        vis_dir: str,
        data_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process video with continuous object ID tracking across frames.

        This implements IMPROVED bidirectional tracking approach:
        1. Sample keyframes every `step` frames
        2. Run Grounding DINO + SAM2 on keyframes to detect objects
        3. Track objects with continuous IDs using IoU matching
        4. Accumulate ALL discovered objects across keyframes
        5. Bidirectional propagation (forward AND reverse) for complete coverage
        6. Favor smaller masks when suppressing overlaps

        Args:
            frames: Video frames (N, H, W, 3)
            description: Text description
            main_object: Main object name
            text_prompt: Text prompt for Grounding DINO
            vis_dir: Directory to save visualizations
            data_dict: Additional data

        Returns:
            Results dictionary with per-frame masks and object tracking info
        """
        print(f"VIDEO INSTANCE SEG MODE: Processing {len(frames)} frames with continuous ID tracking...")

        # Create subdirectories
        frame_dir = os.path.join(vis_dir, "frames")
        mask_data_dir = os.path.join(vis_dir, "mask_data")
        json_data_dir = os.path.join(vis_dir, "json_data")
        result_dir = os.path.join(vis_dir, "result")

        CommonUtils.creat_dirs(frame_dir)
        CommonUtils.creat_dirs(mask_data_dir)
        CommonUtils.creat_dirs(json_data_dir)
        CommonUtils.creat_dirs(result_dir)

        # Save frames to directory for SAM2 video predictor
        print("Saving frames to temporary directory...")
        frame_names = self.video_tracker.save_frames_to_directory(frames, frame_dir)
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Initialize video predictor state
        print("Initializing SAM2 video predictor...")
        inference_state = self.video_tracker.init_state(
            video_path=frame_dir,
            offload_video_to_cpu=True,
            async_loading_frames=True
        )

        # Tracking parameters
        step = 100  # Keyframe sampling interval
        iou_threshold = 0.5  # IoU threshold for matching propagated masks with detected masks

        # Storage for all propagated masks: frame_idx -> {obj_id -> mask_info}
        all_frame_masks = {i: {} for i in range(len(frame_names))}

        # PHASE 1: Sliding window detection and propagation with continuous ID tracking
        print(f"\n=== PHASE 1: Sliding window tracking with continuous IDs (step={step}) ===")

        objects_count = 0
        prev_segment_masks = MaskDictionaryModel()  # Masks propagated to current keyframe from previous segment

        # Process video in overlapping segments
        keyframe_indices = list(range(0, len(frame_names), step))

        for segment_idx in range(len(keyframe_indices)):
            current_keyframe = keyframe_indices[segment_idx]
            print(f"\n--- Segment {segment_idx}: Keyframe {current_keyframe} ---")

            # Load keyframe image
            img_path = os.path.join(frame_dir, frame_names[current_keyframe])
            image = Image.open(img_path)

            # Run Grounding DINO + SAM2 detection on current keyframe
            masks, scores, logits, boxes, labels = self.grounded_sam2(
                np.array(image.convert("RGB")),
                text_prompt
            )

            if len(boxes) == 0:
                print(f"  No objects detected at keyframe {current_keyframe}")
                self.grounded_sam2.reset_predictor()
                # Still propagate previous objects forward if they exist
                if len(prev_segment_masks.labels) > 0:
                    prev_segment_masks = MaskDictionaryModel()
                continue

            print(f"  Detected {len(boxes)} objects at keyframe {current_keyframe}: {labels}")

            # Create mask dictionary for detected objects at current keyframe
            detected_masks = MaskDictionaryModel()
            detected_masks.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(self.video_tracker.device),
                box_list=torch.tensor(boxes),
                label_list=labels
            )

            # Match detected masks with propagated masks from previous segment using IoU
            # This assigns continuous IDs across segments
            objects_count = detected_masks.update_masks(
                tracking_annotation_dict=prev_segment_masks,
                iou_threshold=iou_threshold,
                objects_count=objects_count
            )

            # Merge with undetected objects from previous segment
            # Objects that exist in prev_segment but weren't re-detected should still be propagated
            all_objects_at_keyframe = MaskDictionaryModel()
            all_objects_at_keyframe.labels = copy.deepcopy(detected_masks.labels)

            # Add objects from previous segment that weren't matched (not re-detected)
            matched_prev_ids = set()
            for obj_id, obj_info in detected_masks.labels.items():
                # Check if this obj_id existed in prev_segment_masks
                if obj_id in prev_segment_masks.labels:
                    matched_prev_ids.add(obj_id)

            # Add unmatched previous objects (they still exist but weren't detected this frame)
            for prev_obj_id, prev_obj_info in prev_segment_masks.labels.items():
                if prev_obj_id not in matched_prev_ids:
                    print(f"    Carrying forward undetected object {prev_obj_id} ({prev_obj_info.class_name})")
                    all_objects_at_keyframe.labels[prev_obj_id] = prev_obj_info

            print(f"  Detected IDs: {list(detected_masks.labels.keys())}")
            print(f"  All objects at keyframe (detected + carried forward): {list(all_objects_at_keyframe.labels.keys())}")
            print(f"  Total unique objects so far: {objects_count}")

            # Determine segment boundaries for propagation
            segment_start = current_keyframe
            if segment_idx < len(keyframe_indices) - 1:
                segment_end = keyframe_indices[segment_idx + 1]
            else:
                segment_end = len(frame_names) - 1

            print(f"  Propagating segment: frames {segment_start} to {segment_end}")

            # Bidirectional propagation within this segment
            for obj_id, obj_info in all_objects_at_keyframe.labels.items():
                print(f"    Propagating object {obj_id} ({obj_info.class_name})...")

                # Reset state for this object
                self.video_tracker.reset_state(inference_state)

                # Get mask at current keyframe
                anchor_mask = obj_info.mask

                # Ensure mask has correct dtype
                if anchor_mask.dtype != self.video_tracker.model_dtype:
                    anchor_mask = anchor_mask.to(self.video_tracker.model_dtype)
                if str(anchor_mask.device) != str(self.video_tracker.device):
                    anchor_mask = anchor_mask.to(self.video_tracker.device)

                # Add anchor mask
                self.video_tracker.add_new_mask(
                    inference_state,
                    current_keyframe,
                    obj_id,
                    anchor_mask
                )

                # Backward propagation (current_keyframe → segment_start)
                if current_keyframe > segment_start:
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_tracker.propagate_in_video(
                        inference_state,
                        start_frame_idx=current_keyframe,
                        max_frame_num_to_track=current_keyframe - segment_start + 1,
                        reverse=True
                    ):
                        if segment_start <= out_frame_idx <= current_keyframe:
                            for i, out_obj_id in enumerate(out_obj_ids):
                                if out_obj_id == obj_id:
                                    out_mask = (out_mask_logits[i] > 0.0)[0]
                                    all_frame_masks[out_frame_idx][obj_id] = {
                                        'mask': out_mask,
                                        'class_name': obj_info.class_name,
                                        'mask_size': out_mask.sum().item()
                                    }
                                    break

                # Forward propagation (current_keyframe → segment_end)
                self.video_tracker.reset_state(inference_state)
                self.video_tracker.add_new_mask(
                    inference_state,
                    current_keyframe,
                    obj_id,
                    anchor_mask
                )

                if current_keyframe < segment_end:
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_tracker.propagate_in_video(
                        inference_state,
                        start_frame_idx=current_keyframe,
                        max_frame_num_to_track=segment_end - current_keyframe + 1,
                        reverse=False
                    ):
                        if current_keyframe <= out_frame_idx <= segment_end:
                            for i, out_obj_id in enumerate(out_obj_ids):
                                if out_obj_id == obj_id:
                                    out_mask = (out_mask_logits[i] > 0.0)[0]
                                    all_frame_masks[out_frame_idx][obj_id] = {
                                        'mask': out_mask,
                                        'class_name': obj_info.class_name,
                                        'mask_size': out_mask.sum().item()
                                    }
                                    break

            # Prepare masks for next segment matching
            # Propagate current segment's masks to the next keyframe for IoU matching
            if segment_idx < len(keyframe_indices) - 1:
                next_keyframe = keyframe_indices[segment_idx + 1]
                prev_segment_masks = MaskDictionaryModel()

                # Use the masks we already propagated to next_keyframe
                if next_keyframe in all_frame_masks:
                    for obj_id, obj_info in all_frame_masks[next_keyframe].items():
                        obj_info_model = ObjectInfo(
                            instance_id=obj_id,
                            mask=obj_info['mask'],
                            class_name=obj_info['class_name']
                        )
                        prev_segment_masks.labels[obj_id] = obj_info_model

            # Reset predictor and clean up
            self.grounded_sam2.reset_predictor()
            del masks, scores, logits, boxes, labels, detected_masks
            torch.cuda.empty_cache()
            gc.collect()

        # PHASE 2: Save results
        print(f"\n=== PHASE 2: Saving masks and metadata ===")
        for frame_idx in range(len(frame_names)):
            frame_name = frame_names[frame_idx].split(".")[0]
            frame_masks_dict = all_frame_masks.get(frame_idx, {})

            if len(frame_masks_dict) == 0:
                # Save empty mask
                mask_dict = MaskDictionaryModel(
                    mask_name=f"mask_{frame_name}.npy",
                    mask_height=frames.shape[1],
                    mask_width=frames.shape[2]
                )
                mask_dict.save_empty_mask_and_json(
                    mask_data_dir, json_data_dir,
                    image_name_list=[frame_names[frame_idx]]
                )
            else:
                # Create combined mask image
                mask_img = torch.zeros(frames.shape[1], frames.shape[2])
                frame_mask_model = MaskDictionaryModel(
                    mask_name=f"mask_{frame_name}.npy",
                    mask_height=frames.shape[1],
                    mask_width=frames.shape[2]
                )

                for obj_id, obj_info in frame_masks_dict.items():
                    mask_img[obj_info['mask'] == True] = obj_id

                    # Add to frame mask model
                    obj_info_model = ObjectInfo(
                        instance_id=obj_id,
                        mask=obj_info['mask'],
                        class_name=obj_info['class_name']
                    )
                    obj_info_model.update_box()
                    frame_mask_model.labels[obj_id] = obj_info_model

                # Save mask and JSON
                np.save(os.path.join(mask_data_dir, f"mask_{frame_name}.npy"),
                       mask_img.numpy().astype(np.uint16))

                json_data_path = os.path.join(json_data_dir, f"mask_{frame_name}.json")
                with open(json_data_path, "w") as f:
                    json.dump(frame_mask_model.to_dict(), f)

        # Step 8: Visualize results
        print("Creating visualizations...")
        CommonUtils.draw_masks_and_box_with_supervision(
            frame_dir, mask_data_dir, json_data_dir, result_dir
        )

        # Step 9: Create output video
        from src.utils.video_utils import create_video_from_images
        video_output_path = os.path.join(vis_dir, "tracking_video.mp4")
        create_video_from_images(result_dir, video_output_path, frame_rate=15)

        # Step 10: Build results structure
        results = []
        for frame_idx in range(len(frames)):
            frame_name = f"{frame_idx:05d}"
            mask_path = os.path.join(mask_data_dir, f"mask_{frame_name}.npy")
            json_path = os.path.join(json_data_dir, f"mask_{frame_name}.json")

            frame_result = {
                "frame_idx": frame_idx,
                "main_object": main_object,
                "mask_path": mask_path if os.path.exists(mask_path) else None,
                "json_path": json_path if os.path.exists(json_path) else None,
                "visualization_path": os.path.join(result_dir, f"{frame_name}.jpg")
            }

            # Load mask info if available
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    mask_info = json.load(f)
                    frame_result["objects"] = mask_info.get("labels", {})

            results.append(frame_result)

        # Clean up temporary frame directory
        print("Cleaning up temporary files...")
        shutil.rmtree(frame_dir, ignore_errors=True)

        return {
            "video_name": data_dict["video_name"],
            "description": description,
            "main_object": main_object,
            "total_objects_tracked": objects_count,
            "metadata": data_dict.get("metadata", {}),
            "results": results,
            "video_path": video_output_path
        }

    def _process_tracking_all_objects(
        self,
        frames: np.ndarray,
        description: str,
        main_object: str,
        text_prompt: str,
        vis_dir: str,
        data_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Track all objects mentioned in text prompt.
        Detects objects only in first frame and propagates forward.

        Args:
            frames: Video frames (N, H, W, 3)
            description: Text description
            main_object: Main object name
            text_prompt: Text prompt containing all objects to track
            vis_dir: Directory to save visualizations
            data_dict: Additional data

        Returns:
            Results dictionary with per-frame masks and object tracking info
        """
        print(f"ALL OBJECTS MODE: Tracking all objects in prompt across {len(frames)} frames...")
        print(f"Text prompt: {text_prompt}")

        # Create subdirectories
        frame_dir = os.path.join(vis_dir, "frames")
        mask_data_dir = os.path.join(vis_dir, "mask_data")
        json_data_dir = os.path.join(vis_dir, "json_data")
        result_dir = os.path.join(vis_dir, "result")

        CommonUtils.creat_dirs(frame_dir)
        CommonUtils.creat_dirs(mask_data_dir)
        CommonUtils.creat_dirs(json_data_dir)
        CommonUtils.creat_dirs(result_dir)

        # Save frames to directory
        print("Saving frames to temporary directory...")
        frame_names = self.video_tracker.save_frames_to_directory(frames, frame_dir)
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Initialize video predictor state
        print("Initializing SAM2 video predictor...")
        inference_state = self.video_tracker.init_state(
            video_path=frame_dir,
            offload_video_to_cpu=True,
            async_loading_frames=True
        )

        # Step 1: Detect all objects in first frame only
        print("\n=== Detecting objects in first frame ===")
        first_frame_img_path = os.path.join(frame_dir, frame_names[0])
        first_frame = Image.open(first_frame_img_path)
        masks, scores, logits, boxes, labels = self.grounded_sam2(first_frame, text_prompt)

        if len(boxes) == 0:
            print("No objects detected in first frame!")
            self.grounded_sam2.reset_predictor()
            return {
                "video_name": data_dict["video_name"],
                "description": description,
                "main_object": main_object,
                "text_prompt": text_prompt,
                "total_objects_tracked": 0,
                "metadata": data_dict.get("metadata", {}),
                "results": []
            }

        print(f"Detected {len(boxes)} objects in first frame: {labels}")

        # Storage for all propagated masks
        all_frame_masks = {i: {} for i in range(len(frame_names))}

        # Step 2: Propagate each detected object forward through the video
        print("\n=== Propagating objects forward ===")
        for obj_id, (mask, label) in enumerate(zip(masks, labels)):
            print(f"Propagating object {obj_id} ({label})...")

            # Reset state for this object
            self.video_tracker.reset_state(inference_state)

            # Convert mask to tensor
            anchor_mask = torch.tensor(mask).to(self.video_tracker.device)
            if anchor_mask.dtype != self.video_tracker.model_dtype:
                anchor_mask = anchor_mask.to(self.video_tracker.model_dtype)

            # Add mask at first frame
            self.video_tracker.add_new_mask(
                inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=anchor_mask
            )

            # Propagate forward through all frames
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_tracker.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=len(frame_names),
                reverse=False
            ):
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)[0]

                    # Compute bbox from mask
                    out_mask_np = out_mask.cpu().numpy() if isinstance(out_mask, torch.Tensor) else out_mask
                    bbox = mask_to_bbox(out_mask_np)

                    all_frame_masks[out_frame_idx][obj_id] = {
                        'mask': out_mask,
                        'bbox': bbox,
                        'class_name': label,
                        'mask_size': out_mask.sum().item()
                    }
                    break
        # Clean up
        self.grounded_sam2.reset_predictor()
        del masks, scores, logits, boxes
        torch.cuda.empty_cache()
        gc.collect()

        # Step 3: Save masks and metadata
        print("\n=== Saving masks and metadata ===")
        for frame_idx in range(len(frame_names)):
            frame_name = frame_names[frame_idx].split(".")[0]
            frame_masks_dict = all_frame_masks.get(frame_idx, {})

            if len(frame_masks_dict) == 0:
                # Save empty mask
                mask_dict = MaskDictionaryModel(
                    mask_name=f"mask_{frame_name}.npy",
                    mask_height=frames.shape[1],
                    mask_width=frames.shape[2]
                )
                mask_dict.save_empty_mask_and_json(
                    mask_data_dir, json_data_dir,
                    image_name_list=[frame_names[frame_idx]]
                )
            else:
                # Create combined mask image
                mask_img = torch.zeros(frames.shape[1], frames.shape[2])
                frame_mask_model = MaskDictionaryModel(
                    mask_name=f"mask_{frame_name}.npy",
                    mask_height=frames.shape[1],
                    mask_width=frames.shape[2]
                )

                for obj_id, obj_info in frame_masks_dict.items():
                    mask_img[obj_info['mask'] == True] = obj_id+1

                    # Add to frame mask model
                    obj_info_model = ObjectInfo(
                        instance_id=obj_id+1,
                        mask=obj_info['mask'],
                        class_name=obj_info['class_name']
                    )
                    obj_info_model.update_box()
                    frame_mask_model.labels[obj_id+1] = obj_info_model

                # Save mask and JSON
                np.save(os.path.join(mask_data_dir, f"mask_{frame_name}.npy"),
                       mask_img.numpy().astype(np.uint16))

                json_data_path = os.path.join(json_data_dir, f"mask_{frame_name}.json")
                with open(json_data_path, "w") as f:
                    json.dump(frame_mask_model.to_dict(), f)

        # Step 4: Visualize results
        print("Creating visualizations...")
        CommonUtils.draw_masks_and_box_with_supervision(
            frame_dir, mask_data_dir, json_data_dir, result_dir
        )
        CommonUtils.draw_visual_trace(
            result_dir, json_data_dir, result_dir
        )

        # Step 5: Create output video
        from src.utils.video_utils import create_video_from_images
        video_output_path = os.path.join(vis_dir, "tracking_video.mp4")
        create_video_from_images(result_dir, video_output_path, frame_rate=15)

        # Step 6: Build results structure
        results = []
        for frame_idx in range(len(frames)):
            frame_name = f"{frame_idx:05d}"
            mask_path = os.path.join(mask_data_dir, f"mask_{frame_name}.npy")
            json_path = os.path.join(json_data_dir, f"mask_{frame_name}.json")

            frame_result = {
                "frame_idx": frame_idx,
                "main_object": main_object,
                "mask_path": mask_path if os.path.exists(mask_path) else None,
                "json_path": json_path if os.path.exists(json_path) else None,
                "visualization_path": os.path.join(result_dir, f"{frame_name}.jpg")
            }

            # Load mask info if available
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    mask_info = json.load(f)
                    frame_result["objects"] = mask_info.get("labels", {})

            results.append(frame_result)

        # Clean up temporary files
        print("Cleaning up temporary files...")
        shutil.rmtree(frame_dir, ignore_errors=True)

        return {
            "video_name": data_dict["video_name"],
            "description": description,
            "main_object": main_object,
            "text_prompt": text_prompt,
            "total_objects_tracked": len(labels),
            "metadata": data_dict.get("metadata", {}),
            "results": results,
            "video_path": video_output_path
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
        text_prompt = f"hand. gripper."

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
        elif self.mode == "video_instance_seg" or self.mode == "video_instance_seg_sam3":
            # VIDEO INSTANCE SEG MODE: Continuous ID tracking across frames
            segment_result = self._process_video_mode_with_continuous_id(
                segment_frames, description, main_object, text_prompt, vis_dir, segment_data_dict
            )
        elif self.mode == "all_objects":
            # Track all objects mentioned in text prompt, not just target
            segment_result = self._process_tracking_all_objects(
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
    parser.add_argument(
        "--mode",
        type=str,
        default="all_objects",
        help="Override mode from config"
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

    if args.mode is not None:
        config["mode"] = args.mode

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

from typing import Any, Dict, List
import os
import json
import shutil

import numpy as np
import torch

from src.models.gemma import Gemma
from src.models.sam3_video_tracker import SAM3VideoTracker
from src.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from src.utils.common_utils import CommonUtils

from .base import BasePipeline, load_config


class AffordanceType1Pipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hand_only = config.get('hand_only', False)
        self.text_prompt = config.get('text_prompt', None)
        self.debug = config.get('debug', False)
        self.ema_alpha = config.get('ema_alpha', 0.9)
        self.gemma = None

        # Initialize Gemma if hand only is False to extract main object
        if self.hand_only==False:
            self.gemma = Gemma(**config.get("gemma", {}))


        sam3_config = config.get("sam3", {})
        self.video_tracker = SAM3VideoTracker(
            gpus_to_use=sam3_config.get("gpus_to_use", [0]),
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_id=sam3_config.get("model_id", "facebook/sam3")
        )


    def preprocess(self, data_dict: Dict[str, Any]):
        """Prepare input data for processing."""
        return data_dict


    def _process_visual_trace(
        self,
        frames: np.ndarray,
        description: str,
        vis_dir: str,
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
            
        Returns:
            Results dictionary with per-frame masks and object tracking info
        """
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
        frame_names = self.video_tracker.save_frames_to_directory(frames, frame_dir)
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Initialize video predictor state
        inference_state = self.video_tracker.init_state(
            video_path=frame_dir
        )

        # Step 1: Detect all objects in first frame
        print("\n=== Detecting objects in first frame ===")
        if self.hand_only :
            text_prompt = self.text_prompt if self.text_prompt else 'robot arm'
        else : 
            main_object = self.gemma(description).strip()
            text_prompt = f"{main_object}. hand. gripper."
        print(f"Using text prompt: {text_prompt}")

        frame_idx, obj_ids, outputs = self.video_tracker.add_new_mask_with_text(
            inference_state,
            frame_idx=0,
            text_prompt=text_prompt
        )
        if len(obj_ids) == 0:
            print("No objects detected in first frame!")
            self.video_tracker.close_session(inference_state)
            return {
                "description": description,
                "text_prompt": text_prompt,
                "total_objects_tracked": 0,
                "results": []
            }
        labels = [f"object_{obj_info['id']}" for obj_info in outputs]
        print(f"Detected {len(obj_ids)} objects in first frame: {labels}")
    
        # Step 2: Propagate objects forward
        print("\n=== Propagating objects forward ===")
        all_frame_masks = self.video_tracker.propagate_all_objects(
            inference_state,
            start_frame=0,
            end_frame=len(frame_names) - 1,
            propagation_direction="forward"
        )

        # Step 3: Build frame mask models and compute centroids
        print("\n=== Saving masks and metadata ===")
        frame_mask_models = {}  # {frame_idx: MaskDictionaryModel}
        mask_images = {}  # {frame_idx: mask_img tensor}

        for frame_idx in range(len(frame_names)):
            frame_name = frame_names[frame_idx].split(".")[0]
            frame_masks_dict = all_frame_masks.get(frame_idx, {})
            if len(frame_masks_dict) == 0:
                # Save empty mask
                mask_dict = MaskDictionaryModel(
                    mask_name=f"mask_{frame_name}",
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
                    mask_name=f"mask_{frame_name}",
                    mask_height=frames.shape[1],
                    mask_width=frames.shape[2]
                )

                for obj_id, obj_info in frame_masks_dict.items():
                    mask_img[obj_info['mask'] == True] = obj_id+1
                    obj_info_model = ObjectInfo(
                        instance_id=obj_id+1,
                        mask=obj_info['mask'],
                        class_name=obj_info['class_name']
                    )

                    bbox = obj_info.get('bbox', None)
                    img_width = frames.shape[2]
                    img_height = frames.shape[1]
                    if bbox is not None:
                        # SAM3 returns normalized coordinates [x, y, w, h] in range [0, 1]
                        # Convert to pixel coordinates and then to [x_min, y_min, x_max, y_max]
                        bbox = [
                            int(bbox[0] * img_width),  # x_min
                            int(bbox[1] * img_height),  # y_min
                            int((bbox[0] + bbox[2]) * img_width),  # x_max = (x + width) * img_width
                            int((bbox[1] + bbox[3]) * img_height)  # y_max = (y + height) * img_height
                        ]
                    obj_info_model.update_box(bbox)
                    frame_mask_model.labels[obj_id+1] = obj_info_model

                frame_mask_models[frame_idx] = frame_mask_model
                mask_images[frame_idx] = mask_img

        # Step 3.5: Apply EMA smoothing to centroids
        print("\n=== Computing EMA centroids ===")
        prev_ema = {}  # {obj_id: (ema_cx, ema_cy)}
        alpha = self.ema_alpha

        for frame_idx in sorted(frame_mask_models.keys()):
            frame_mask_model = frame_mask_models[frame_idx]
            for obj_id, obj_info in frame_mask_model.labels.items():
                cx, cy = obj_info.centroid_x, obj_info.centroid_y

                if obj_id in prev_ema:
                    ema_cx = alpha * cx + (1 - alpha) * prev_ema[obj_id][0]
                    ema_cy = alpha * cy + (1 - alpha) * prev_ema[obj_id][1]
                else:
                    ema_cx, ema_cy = cx, cy  # First frame: EMA = raw

                obj_info.ema_centroid_x = ema_cx
                obj_info.ema_centroid_y = ema_cy
                prev_ema[obj_id] = (ema_cx, ema_cy)

        # Step 3.6: Save JSON and mask files
        for frame_idx, frame_mask_model in frame_mask_models.items():
            frame_name = frame_names[frame_idx].split(".")[0]
            json_data_path = os.path.join(json_data_dir, f"mask_{frame_name}.json")
            with open(json_data_path, "w") as f:
                json.dump(frame_mask_model.to_dict(), f)
            np.save(os.path.join(mask_data_dir, f"mask_{frame_name}.npy"),
                mask_images[frame_idx].numpy().astype(np.uint16))

        # Step 4: Visualize results (only if debug mode is enabled)
        if self.debug:
            print("Creating visualizations...")
            CommonUtils.draw_masks_and_box_with_supervision(
                frame_dir, mask_data_dir, json_data_dir, result_dir
            )
            CommonUtils.draw_visual_trace(
                result_dir, json_data_dir, result_dir
            )
            from src.utils.video_utils import create_video_from_images
            video_output_path = os.path.join(vis_dir, "tracking_video.mp4")
            create_video_from_images(result_dir, video_output_path, frame_rate=15)
        else:
            print("Skipping visualizations (debug mode disabled)")

        # Step 5 : Save results
        results = []
        for frame_idx in range(len(frames)):
            frame_name = f"{frame_idx:05d}"
            json_path = os.path.join(json_data_dir, f"mask_{frame_name}.json")
            frame_result = {
                "frame_idx": frame_idx,
                "json_path": json_path if os.path.exists(json_path) else None
            }
            results.append(frame_result)

        # Clean up temporary files
        shutil.rmtree(frame_dir, ignore_errors=True)
        
        return {
            "description": description,
            "text_prompt": text_prompt,
            "total_objects_tracked": len(labels),
            "results": results,
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
        
        # Create save directory for visualizations
        save_dir = self.config.get("save_dir", ".")
        output_dir = self.config.get("output_dir", "output")
        sanitized_video_name = video_name.replace("/", "_")

        if dataset_name:
            vis_dir = os.path.join(save_dir, output_dir, dataset_name, sanitized_video_name, f"segment_{segment_idx}")
        else:
            vis_dir = os.path.join(save_dir, output_dir, sanitized_video_name, f"segment_{segment_idx}")
        os.makedirs(vis_dir, exist_ok=True)

        segment_result = self._process_visual_trace(
            segment_frames, description, vis_dir
        )

        # Add segment metadata
        segment_result.update({
            "segment_idx": segment_idx,
            "start_frame": start_frame,
            "end_frame": end_frame
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
            "num_segments": len(all_segments),
            "segments": all_segments
        }


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test AffordanceType1 pipeline")
    parser.add_argument(
        "-s", "--segment-index",
        type=int,
        default=None,
        help="Process only a specific segment index. If not provided, processes all segments (default: None)"
    )
    args = parser.parse_args()

    # Load pipeline configuration and create pipeline
    config = load_config("affordance_type1")

    from src.datasets.agibotworld import AgiBotWorldDataset
    ds = AgiBotWorldDataset()
    data_dict = ds[1]
    if args.segment_index is not None:
        if args.segment_index >= len(data_dict['descriptions']):
            print(f"Error: Segment index {args.segment_index} out of range (0-{len(data_dict['descriptions'])-1})")
            exit(1)

        print(f"\nProcessing only segment {args.segment_index}")
        # Keep only the selected segment
        data_dict['descriptions'] = [data_dict['descriptions'][args.segment_index]]

    pipeline = AffordanceType1Pipeline(config)
    
    # Run pipeline
    print("\nRunning affordance type1 pipeline...")
    results = pipeline(data_dict, save_dir=".")

    # Save results to JSON
    dataset_name = config.get("dataset", {}).get("name", "unknown")
    output_dir = config.get("output_dir", "visualizations")
    sanitized_video_name = data_dict['video_name'].replace("/", "_")
    results_dir = os.path.join(".", output_dir, dataset_name, sanitized_video_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved pipeline results to {results_path}")


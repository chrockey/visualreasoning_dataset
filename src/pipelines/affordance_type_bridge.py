from typing import Any, Dict, List, Optional, Tuple
import os
import json
import shutil

import numpy as np
import torch

from src.models.sam3_video_tracker import SAM3VideoTracker
from src.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from src.utils.common_utils import CommonUtils
import ast
from src.datasets.oxe import OXEDataset
from .base import BasePipeline, load_config
import google.generativeai as genai
from PIL import Image
import re

# ============================================================
# OXE -> Pipeline input adapter
# ============================================================
def _pick_first_nonempty_str(cands: List[Any], default: str = "") -> str:
    for x in cands:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return default


def oxe_sample_to_affordance_input(
    sample: Dict[str, Any],
    episode_index: int,
) -> Dict[str, Any]:
    """
    Convert OXEDataset __getitem__ output -> AffordanceType1Pipeline expected dict.
    Expected by pipeline:
      - frames: (T,H,W,3) uint8 RGB
      - descriptions: List[(start_frame, end_frame, description_text)]
      - video_name: str
      - metadata: dict
    """

    # 1) frames
    # Many implementations return either sample["frames"] or sample["images"] etc.
    frames = None
    for k in ["frames", "images", "rgb", "video", "observations"]:
        if k in sample:
            frames = sample[k]
            break
    if frames is None:
        raise KeyError("OXE sample does not contain frames-like key (frames/images/rgb/video/observations).")

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3). Got {frames.shape}")

    # ensure uint8
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    T = frames.shape[0]

    # 2) description text
    # OXE sometimes stores language per step or per episode.
    # Try common fields:
    desc = ""
    # (a) top-level keys
    desc = _pick_first_nonempty_str([
        sample.get("description", ""),
        sample.get("instruction", ""),
        sample.get("language_instruction", ""),
        sample.get("task", ""),
    ], default="")

    # (b) metadata keys
    md = sample.get("metadata", {}) if isinstance(sample.get("metadata", {}), dict) else {}
    desc = _pick_first_nonempty_str([
        desc,
        md.get("language_instruction", ""),
        md.get("instruction", ""),
        md.get("task", ""),
        md.get("text", ""),
    ], default="")

    # (c) per-step language list
    if not desc:
        for k in ["language_instructions", "instructions", "texts"]:
            if k in sample and isinstance(sample[k], (list, tuple)) and len(sample[k]) > 0:
                desc = _pick_first_nonempty_str([sample[k][0]], default="")
                break


    # 3) single segment over entire episode by default
    descriptions = [(0, T - 1, desc)]

    # 4) video name
    # dataset name may exist in metadata
    dataset_name = _pick_first_nonempty_str([
        md.get("dataset_name", ""),
        md.get("dataset", ""),
        sample.get("dataset_name", ""),
        sample.get("dataset", ""),
    ], default="oxe")

    video_name = f"{dataset_name}/episode_{episode_index:05d}"

    # 5) metadata: keep original metadata if any
    metadata = md.copy()
    metadata["episode_index"] = episode_index
    metadata["dataset_name"] = dataset_name

    return {
        "frames": frames,
        "descriptions": descriptions,
        "video_name": video_name,
        "metadata": metadata,
    }


# ============================================================
# Pipeline
# ============================================================
class AffordanceType1Pipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hand_only = config.get("hand_only", False)
        self.text_prompt1 = config.get("text_prompt1", None)
        self.text_prompt2 = config.get("text_prompt2", None)
        self.debug = config.get("debug", False)
        self.ema_alpha = config.get("ema_alpha", 0.9)

        # SAM3
        sam3_config = config.get("sam3", {})
        self.video_tracker = SAM3VideoTracker(
            gpus_to_use=sam3_config.get("gpus_to_use", [0]),
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_id=sam3_config.get("model_id", "facebook/sam3"),
        )

        # Gemini
        gemini_cfg = config.get("gemini", {})
        api_key = gemini_cfg.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        self.gemini_model_name = gemini_cfg.get("model_name", "")

        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Set config['gemini']['api_key'] or the GEMINI_API_KEY env var."
            )

        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(self.gemini_model_name)

    def preprocess(self, data_dict: Dict[str, Any]):
        return data_dict
    
    @staticmethod
    def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
        """Convert absolute coordinates to relative coordinates (0-1 range)

        Args:
            coords: List of coordinates
            coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
        """
        if coord_type == "point":
            return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
        elif coord_type == "box":
            return [[
                [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
                for x, y, w, h in coords
            ]]
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")



    def _process_visual_trace(
        self,
        frames: np.ndarray,
        description: str,
        vis_dir: str,
    ) -> Dict[str, Any]:
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
        inference_state = self.video_tracker.init_state(video_path=frame_dir)

        # Step 1: Detect the robot gripper in the first frame using Gemini
        # -----------------------------------------------------------
        print("\n=== Detecting objects in first frame ===")
        
        text_prompt1 = self.text_prompt1 if self.text_prompt1 else "Robot Arm"
        text_prompt2 = self.text_prompt2 if self.text_prompt2 else "Black Object"
        
        frame0 = frames[0]
        IMG_HEIGHT, IMG_WIDTH = frame0.shape[:2]
        pil_image = Image.fromarray(frame0)




        frame_idx, obj_ids, outputs = self.video_tracker.add_new_mask_with_text(
            inference_state,
            frame_idx=0,
            text_prompt=text_prompt1,            
        )
        if len(obj_ids) == 0:
            frame_idx, obj_ids, outputs = self.video_tracker.add_new_mask_with_text(
            inference_state,
            frame_idx=0,
            text_prompt=text_prompt2,            
        )
            
        out_probs = outputs['out_probs']
        out_box = outputs['out_boxes_xywh']
        best_idx = int(np.argmax(out_probs))

        # --- [Gemini Integration Start] ---
        best_box_xywh = out_box[best_idx]  # [x, y, w, h] normalized

        gemini_prompt = (
            "You are a precise robotic vision bounding-box annotator.\n"
            "Your output will be parsed by a program.\n"
            "\n"
            "Task:\n"
            "- Detect ONLY the robot gripper (end-effector) in this image.\n"
            "- Return EXACTLY ONE bounding box.\n"
            "\n"
            "IMPORTANT ROI CONSTRAINT (FROM TRACKER):\n"
            "- You are given a candidate bounding box from a tracker.\n"
            "- The box is in NORMALIZED coordinates [x, y, w, h] relative to the full image.\n"
            "- You MUST search for the gripper ONLY inside this box.\n"
            "- You MUST NOT use any visual evidence outside this box.\n"
            "\n"
            "Tracker-provided candidate box (normalized xywh):\n"
            f"- x = {best_box_xywh[0]}\n"
            f"- y = {best_box_xywh[1]}\n"
            f"- w = {best_box_xywh[2]}\n"
            f"- h = {best_box_xywh[3]}\n"
            "\n"
            "Refinement rules:\n"
            "- The tracker box may be loose or include non-gripper regions.\n"
            "- Within the box ONLY, identify the region that truly corresponds to the gripper.\n"
            "- Return a tighter bounding box around the gripper.\n"
            "- The final bounding box MUST be fully inside the tracker-provided box.\n"
            "\n"
            "Definition of 'gripper' (include/exclude):\n"
            "- Include: end-effector housing/body, both fingers/jaws, tips/pads.\n"
            "- Exclude: robot arm links, wrist joint/flange, forearm, cables, bowl, table, background objects.\n"
            "\n"
            "Output coordinate system:\n"
            "- Output coordinates MUST be in PIXEL units.\n"
            "- The normalized tracker box is ONLY for constraining the search region.\n"
            "\n"
            "Image info:\n"
            "- Resolution: {H} (height) x {W} (width) pixels.\n"
            "\n"
            "Return format (MANDATORY):\n"
            "- Output MUST be EXACTLY a Python list on a SINGLE LINE: [x, y, w, h]\n"
            "- x,y: TOP-LEFT corner in pixels\n"
            "- w,h: width and height in pixels\n"
            "- All values MUST be integers\n"
            "\n"
            "STRICT OUTPUT RULE:\n"
            "- Output ONLY the list like [x, y, w, h].\n"
            "- Do NOT output any other words, sentences, reasoning, or formatting.\n"
            "- The very first character must be '[' and the very last character must be ']'.\n"
            "\n"
            "Example (format only):\n"
            "[340, 120, 170, 140]"
        ).format(H=IMG_HEIGHT, W=IMG_WIDTH)


        try:
            resp = self.gemini_model.generate_content([gemini_prompt, pil_image])
            bbox_text = (getattr(resp, "text", None) or "").strip()
            matches = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", bbox_text)
            x, y, w, h = map(int, matches[-1])   
            xywh = [x, y, w, h]
            xywh = ast.literal_eval(bbox_text)
            final_bbox_rel_xywh = self.abs_to_rel_coords(
            [xywh],
            IMG_WIDTH=IMG_WIDTH,
            IMG_HEIGHT=IMG_HEIGHT,
            coord_type="box"
             )[0]

        except Exception as e:
            print(f"Error calling Gemini API: {e}")

        frame_idx, obj_ids, outputs = self.video_tracker.add_new_mask_with_text_bounding_box(
            inference_state,
            frame_idx=0,
            text_prompt=None,
            bounding_boxes = final_bbox_rel_xywh,
            bounding_box_labels= [1]
            
        )

        # Step 2: Propagate objects forward
        print("\n=== Propagating objects forward ===")
        all_frame_masks = self.video_tracker.propagate_all_objects(
            inference_state,
            start_frame=0,
            end_frame=len(frame_names) - 1,
            propagation_direction="forward",
        )

        
        # Step 3: Save masks and metadata
        print("\n=== Saving masks and metadata ===")
        prev_ema = {} 
        prev_bbox_area = {}

        for frame_idx in range(len(frame_names)):
            frame_name = frame_names[frame_idx].split(".")[0]
            frame_masks_dict = all_frame_masks.get(frame_idx, {})

            if len(frame_masks_dict) == 0:
                mask_dict = MaskDictionaryModel(
                    mask_name=f"mask_{frame_name}",
                    mask_height=frames.shape[1],
                    mask_width=frames.shape[2],
                )
                mask_dict.save_empty_mask_and_json(
                    mask_data_dir,
                    json_data_dir,
                    image_name_list=[frame_names[frame_idx]],
                )
            else:
                mask_img = torch.zeros(frames.shape[1], frames.shape[2])
                frame_mask_model = MaskDictionaryModel(
                    mask_name=f"mask_{frame_name}",
                    mask_height=frames.shape[1],
                    mask_width=frames.shape[2],
                )

                for obj_id, obj_info in frame_masks_dict.items():
                    mask_img[obj_info["mask"] == True] = obj_id + 1

                    obj_info_model = ObjectInfo(
                        instance_id=obj_id + 1,
                        mask=obj_info["mask"],
                        class_name=obj_info["class_name"],
                    )

                    bbox = obj_info.get("bbox", None)
                    if bbox is not None:
                        img_width = frames.shape[2]
                        img_height = frames.shape[1]
                        bbox = [
                            int(bbox[0] * img_width),
                            int(bbox[1] * img_height),
                            int((bbox[0] + bbox[2]) * img_width),
                            int((bbox[1] + bbox[3]) * img_height),
                        ]
                        bbox_w = max(0, bbox[2] - bbox[0])
                        bbox_h = max(0, bbox[3] - bbox[1])
                        curr_area = bbox_w * bbox_h
                        
                        if obj_id in prev_bbox_area:
                            prev_area = prev_bbox_area[obj_id]
                            if prev_area > 0 and curr_area >= 2 * prev_area:
                                raise RuntimeError(
                                    f"[BBox Exception Detected] "
                                    f"frame={frame_idx}, obj_id={obj_id}, "
                                    f"prev_area={prev_area}, curr_area={curr_area}"
                                )
                            
                        prev_bbox_area[obj_id] = curr_area
                    obj_info_model.update_box(bbox)

                    # EMA centroid
                    cx, cy = obj_info_model.centroid_x, obj_info_model.centroid_y
                    if obj_id in prev_ema:
                        ema_cx = self.ema_alpha * cx + (1 - self.ema_alpha) * prev_ema[obj_id][0]
                        ema_cy = self.ema_alpha * cy + (1 - self.ema_alpha) * prev_ema[obj_id][1]
                    else:
                        ema_cx, ema_cy = cx, cy
                    obj_info_model.ema_centroid_x = ema_cx
                    obj_info_model.ema_centroid_y = ema_cy
                    prev_ema[obj_id] = (ema_cx, ema_cy)

                    frame_mask_model.labels[obj_id + 1] = obj_info_model

                json_data_path = os.path.join(json_data_dir, f"mask_{frame_name}.json")
                with open(json_data_path, "w") as f:
                    json.dump(frame_mask_model.to_dict(), f)

                np.save(
                    os.path.join(mask_data_dir, f"mask_{frame_name}.npy"),
                    mask_img.numpy().astype(np.uint16),
                )

        # Step 4: Optional visualization
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
            create_video_from_images(result_dir, video_output_path, frame_rate=5)
        else:
            print("Skipping visualizations (debug mode disabled)")

        # Step 5: Save results paths
        results = []
        for frame_idx in range(len(frames)):
            frame_name = f"{frame_idx:05d}"
            json_path = os.path.join(json_data_dir, f"mask_{frame_name}.json")
            results.append({
                "frame_idx": frame_idx,
                "json_path": json_path if os.path.exists(json_path) else None,
            })

        shutil.rmtree(frame_dir, ignore_errors=True)

        return {
            "description": description,
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
        dataset_name: str = None,
    ) -> Dict[str, Any]:
        segment_frames = frames[start_frame : end_frame + 1]

        print(
            f"\nProcessing segment {segment_idx}: [{start_frame}:{end_frame}] (inclusive) "
            f"with {len(segment_frames)} frames"
        )
        print(f"Description: {description}")

        save_dir = self.config.get("save_dir", ".")
        output_dir = self.config.get("output_dir", "output")
        sanitized_video_name = video_name.replace("/", "_")

        if dataset_name:
            vis_dir = os.path.join(
                save_dir, output_dir, dataset_name, sanitized_video_name, f"segment_{segment_idx}"
            )
        else:
            vis_dir = os.path.join(save_dir, output_dir, sanitized_video_name, f"segment_{segment_idx}")

        os.makedirs(vis_dir, exist_ok=True)

        segment_result = self._process_visual_trace(segment_frames, description, vis_dir)

        segment_result.update(
            {"segment_idx": segment_idx, "start_frame": start_frame, "end_frame": end_frame}
        )
        return segment_result

    def process(self, data_dict: Dict[str, Any]):
        frames = data_dict["frames"]
        descriptions = data_dict["descriptions"]
        video_name = data_dict["video_name"]
        metadata = data_dict.get("metadata", {})

        dataset_name = self.config.get("dataset", {}).get("name", None)

        print(f"Processing video: {video_name}")
        print(f"Total segments: {len(descriptions)}")

        all_segments = []
        for segment_idx, (start_frame, end_frame, description) in enumerate(descriptions):
            segment_result = self._process_segment(
                frames, segment_idx, start_frame, end_frame,
                description, video_name, metadata, dataset_name
            )
            all_segments.append(segment_result)

        return {"video_name": video_name, "num_segments": len(all_segments), "segments": all_segments}


# ============================================================
# Main: run on OXE
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AffordanceType1 pipeline on OXE")
    parser.add_argument("--data-dir", type=str, default="./bridge_folder",
                        help="OXE data root dir (your TFRecord/TFDS prepared dir)")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index in OXEDataset")
    parser.add_argument("-s", "--segment-index", type=int, default=None,
                        help="If provided, keep only that segment (after conversion; usually only 1).")
    args = parser.parse_args()

    config = load_config("affordance_bridge")

    # 1) load OXE sample
    ds = OXEDataset(args.data_dir)
    sample = ds[args.episode_index]

    # 2) convert -> pipeline input
    data_dict = oxe_sample_to_affordance_input(
        sample=sample,
        episode_index=args.episode_index
    )

    # optional segment slicing (usually only one segment: [0, T-1])
    if args.segment_index is not None:
        if args.segment_index >= len(data_dict["descriptions"]):
            print(f"Error: segment index {args.segment_index} out of range (0-{len(data_dict['descriptions'])-1})")
            raise SystemExit(1)
        data_dict["descriptions"] = [data_dict["descriptions"][args.segment_index]]

    pipeline = AffordanceType1Pipeline(config)

    print("\nRunning affordance type1 pipeline on OXE...")
    results = pipeline(data_dict, save_dir=".")

    # save results
    dataset_name = config.get("dataset", {}).get("name", "oxe")
    output_dir = config.get("output_dir", "visualizations")
    sanitized_video_name = data_dict["video_name"].replace("/", "_")

    results_dir = os.path.join(".", output_dir, dataset_name, sanitized_video_name)
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved pipeline results to {results_path}")

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import shutil

import numpy as np
import torch

from src.models.sam3_video_tracker import SAM3VideoTracker
from src.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from src.utils.common_utils import CommonUtils

from src.datasets.oxe import OXEDataset
from ..base import BasePipeline, load_config


# ============================================================
# OXE -> Pipeline input adapter
# ============================================================
def _pick_first_nonempty_str(cands: List[Any], default: str = "") -> str:
    for x in cands:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return default

def bottom_contact_midpoint(mask: torch.Tensor, thr: float = 0.5):
    """
    mask: (H, W) torch tensor or numpy array (bool/0-1/float)
    return: (cx, cy) in pixel coords where
      - cy is the lowest y (max y) where mask exists
      - cx is midpoint between leftmost and rightmost mask pixels at that cy
    """
    if mask is None:
        return None, None

    if isinstance(mask, np.ndarray):
        m = torch.from_numpy(mask)
    else:
        m = mask

    # squeeze possible extra dims
    if m.dim() == 4:
        m = m.squeeze(0).squeeze(0)
    elif m.dim() == 3:
        m = m.squeeze(0)

    # binarize
    m = (m > thr)

    ys, xs = torch.where(m)
    if xs.numel() == 0:
        return None, None

    # lowest y (max)
    y_max = int(ys.max().item())

    # all x positions on that lowest row
    xs_bottom = xs[ys == y_max]
    if xs_bottom.numel() == 0:
        return None, None

    x_left = int(xs_bottom.min().item())
    x_right = int(xs_bottom.max().item())

    cx = 0.5 * (x_left + x_right)
    cy = float(y_max)
    return float(cx), float(cy)



def oxe_sample_to_pipeline_input(
    sample: Dict[str, Any],
    episode_index: int,
    default_prompt: str = "Gray object",
) -> Dict[str, Any]:
    """
    Convert OXEDataset __getitem__ output -> SAM3TrackerPipeline expected dict.
    Expected by pipeline:
      - frames: (T,H,W,3) uint8 RGB
      - descriptions: List[(start_frame, end_frame, description_text)]
      - video_name: str
      - metadata: dict
    """

    # 1) frames
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
class SAM3TrackerPipeline(BasePipeline):
    """
    Visual trace pipeline using SAM3 video tracking.
    Dataset-agnostic pipeline that tracks objects using text prompts.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.text_prompt = config.get("text_prompt", None)
        self.debug = config.get("debug", False)
        self.ema_alpha = config.get("ema_alpha", 0.9)
        sam3_config = config.get("sam3", {})
        self.video_tracker = SAM3VideoTracker(
            gpus_to_use=sam3_config.get("gpus_to_use", [0]),
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_id=sam3_config.get("model_id", "facebook/sam3"),
        )

    def preprocess(self, data_dict: Dict[str, Any]):
        return data_dict

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

        # Step 1: Detect all objects in first frame
        print("\n=== Detecting objects in first frame ===")
        text_prompt = self.text_prompt if self.text_prompt else "Gray object"


        frame_idx, obj_ids, outputs = self.video_tracker.add_new_mask_with_text(
            inference_state,
            frame_idx=0,
            text_prompt=text_prompt,
        )

        if len(obj_ids) == 0:
            print("No objects detected in first frame!")
            self.video_tracker.close_session(inference_state)
            return {
                "description": description,
                "text_prompt": text_prompt,
                "total_objects_tracked": 0,
                "results": [],
            }


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
        prev_ema = {}  # {obj_id: (ema_cx, ema_cy)}

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
                    cx, cy = bottom_contact_midpoint(obj_info["mask"], thr=0.5)


                    if cx is None or cy is None:
                        if obj_id in prev_ema:
                            ema_cx, ema_cy = prev_ema[obj_id]
                        else:
                            ema_cx, ema_cy = None, None
                    else:
                        if obj_id in prev_ema and prev_ema[obj_id][0] is not None:
                            ema_cx = self.ema_alpha * cx + (1 - self.ema_alpha) * prev_ema[obj_id][0]
                            ema_cy = self.ema_alpha * cy + (1 - self.ema_alpha) * prev_ema[obj_id][1]
                        else:
                            ema_cx, ema_cy = cx, cy

                        prev_ema[obj_id] = (ema_cx, ema_cy)



                    # EMA centroid
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
            "text_prompt": text_prompt,
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

    parser = argparse.ArgumentParser(description="Run SAM3 Tracker pipeline on OXE")
    parser.add_argument("--config", type=str, required=True,
                        help="Config file name (e.g., visual_trace_language_table)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="OXE data root dir (your TFRecord/TFDS prepared dir)")
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index in OXEDataset")
    parser.add_argument("-s", "--segment-index", type=int, default=None,
                        help="If provided, keep only that segment (after conversion; usually only 1).")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1) load OXE sample
    ds = OXEDataset(args.data_dir)
    sample = ds[args.episode_index]

    # 2) convert -> pipeline input
    data_dict = oxe_sample_to_pipeline_input(
        sample=sample,
        episode_index=args.episode_index,
        default_prompt="Gray object",
    )

    # optional segment slicing (usually only one segment: [0, T-1])
    if args.segment_index is not None:
        if args.segment_index >= len(data_dict["descriptions"]):
            print(f"Error: segment index {args.segment_index} out of range (0-{len(data_dict['descriptions'])-1})")
            raise SystemExit(1)
        data_dict["descriptions"] = [data_dict["descriptions"][args.segment_index]]

    pipeline = SAM3TrackerPipeline(config)

    print("\nRunning SAM3 Tracker pipeline on OXE...")
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

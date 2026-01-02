from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2
from .base import BasePipeline, load_config
from ..utils.gt_visual_trace import (
    draw_sliding_trajectory,
    load_intrinsic_from_dict,
    load_extrinsic_sequence_from_list,
    project_camera_trajectories_to_2d,
    generate_trajectory_visualization_video,
    generate_camera_trajectory_video,
)


class GTVisualTracePipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any], verbose: bool = True, suffix: str = "base"):
        super().__init__(config)
        self.verbose = verbose

        # Camera trajectory visualization config
        self.trace_window = config.get("camera_trajectory", {}).get("trace_window", 60)
        self.viz_width = config.get("camera_trajectory", {}).get("width", 640)
        self.viz_height = config.get("camera_trajectory", {}).get("height", 480)
        self.output_dir = config.get("camera_trajectory", {}).get("output_dir", f"viz/camera_trajectory_{suffix}")

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        """
        Process video and generate camera trajectory visualization.

        Args:
            data_dict: Dictionary containing video data with keys:
                - "frames": (T, H, W, 3) RGB video frames
                - "metadata": Dictionary with camera_params
                - "video_name": Name of the video

        Returns:
            Dictionary containing projected 2D trajectories:
            {
                "head_2d": (T, 2) array,       # Head camera trajectory in pixel coordinates
                "left_hand_2d": (T, 2) array,  # Left hand trajectory in pixel coordinates
                "right_hand_2d": (T, 2) array, # Right hand trajectory in pixel coordinates
                "metadata": {
                    "num_frames": int,          # Number of frames
                    "video_width": int,         # Video width in pixels
                    "video_height": int,        # Video height in pixels
                    "format": str,              # "uv_coordinates"
                    "nan_meaning": str,         # "invisible" (out of bounds or behind camera)
                    "coordinate_system": str    # "image" (top-left origin, x-right, y-down)
                }
            }
            Returns None if camera parameters are missing.
        """
        video_frames = data_dict["frames"]
        metadata = data_dict.get("metadata", {})
        data_name = data_dict.get("video_name", "unknown")

        if self.verbose:
            print(f"[INFO] Processing video: {data_name}")
            print(f"[INFO] Total frames: {len(video_frames)}")
            print(f"[INFO] Step 1: Projecting 3D trajectories to 2D...")

        trajectory_data = project_camera_trajectories_to_2d(
            metadata=metadata,
            num_frames=len(video_frames),
            viz_width=self.viz_width,
            viz_height=self.viz_height,
        )

        if trajectory_data is None:
            print(f"[WARNING] Failed to project trajectories (missing camera parameters)")
            return None

        if self.verbose:
            T = trajectory_data['metadata']['num_frames']
            print(f"[INFO] Successfully projected {T} frames to 2D pixel coordinates")
            print(f"[INFO]   - head_2d: {trajectory_data['head_2d'].shape}")
            print(f"[INFO]   - left_hand_2d: {trajectory_data['left_hand_2d'].shape}")
            print(f"[INFO]   - right_hand_2d: {trajectory_data['right_hand_2d'].shape}")
            print(f"[INFO] Step 2: Generating visualization video from 2D trajectories...")

            # Prepare output directory
            output_dir = Path(self.output_dir) / data_name
            output_dir.mkdir(parents=True, exist_ok=True)
            traj_output_path = str(output_dir / "robot_trajectory.mp4")

            generate_trajectory_visualization_video(
                video_frames=video_frames,
                trajectory_data=trajectory_data,
                output_path=traj_output_path,
                trace_window=self.trace_window,
            )

            print(f"[INFO] Finished processing video {data_name}")

        return trajectory_data


if __name__ == "__main__":
    from src.datasets.agibotworld import AgiBotWorldDataset
    ds = AgiBotWorldDataset("/data01/junmyeong/rlwrld_robotics/robot_data/AgiBotWorld-Beta")

    # get first video
    data_dict = ds[0]

    # load config
    config = load_config("visual_trace")
    pipeline = GTVisualTracePipeline(config, verbose=True)

    # run pipeline
    results = pipeline.process(data_dict)

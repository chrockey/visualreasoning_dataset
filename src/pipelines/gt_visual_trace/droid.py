from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
import cv2

from ..base import BasePipeline, load_config
from ...utils.gt_visual_trace import (
    project_camera_trajectories_to_2d,
    generate_trajectory_visualization_video,
)


class GTVisualTraceDroidPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any], verbose: bool = True, suffix: str = "base"):
        super().__init__(config)
        self.verbose = verbose
        self.trace_window = config.get("camera_trajectory", {}).get("trace_window", 60)
        self.output_dir = config.get("camera_trajectory", {}).get("output_dir", f"viz/camera_trajectory_{suffix}")
        self.droid_cfg = config.get("droid_gt_trace", None)

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    # ------------------------------------------------------------
    # DROID helpers
    # ------------------------------------------------------------
    def _read_mp4_frames_rgb(self, mp4_path: Path) -> np.ndarray:
        """Read an MP4 file into a (T,H,W,3) RGB numpy array."""
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")

        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded from: {mp4_path}")

        return np.stack(frames, axis=0)

    def _find_episode_dir_for_mp4(self, base_dir: Path, mp4_id: str) -> Optional[Path]:
        """
        Search for an episode directory that contains:
          - trajectory.h5
          - recordings/MP4/{mp4_id}.mp4
          - recordings/SVO/{mp4_id}.svo
        """
        if not base_dir.exists():
            return None

        candidates = list(base_dir.rglob(f"{mp4_id}.mp4"))
        for mp4_path in candidates:
            # Expect .../<episode>/recordings/MP4/{id}.mp4
            ep_dir = mp4_path.parent.parent.parent  # MP4 -> recordings -> episode
            traj_h5 = ep_dir / "trajectory.h5"
            svo_path = ep_dir / "recordings" / "SVO" / f"{mp4_id}.svo"
            if traj_h5.exists() and svo_path.exists():
                return ep_dir

        return None

    def _resolve_droid_assets(self, mp4_id: str, episode_dir: Optional[Path]) -> Tuple[Path, Path, Path, str]:
        """
        Resolve (trajectory.h5, mp4_path, svo_path, episode_name) for DROID layout.
        """
        if self.droid_cfg is None:
            raise ValueError("droid_gt_trace config is required to resolve DROID assets")

        base_dir = Path(self.droid_cfg["base_dir"])
        if episode_dir is None:
            episode_dir = self._find_episode_dir_for_mp4(base_dir, mp4_id)

        if episode_dir is None:
            raise FileNotFoundError(f"Episode dir not found for mp4_id={mp4_id} under base_dir={base_dir}")

        traj_h5 = episode_dir / "trajectory.h5"
        mp4_path = episode_dir / "recordings" / "MP4" / f"{mp4_id}.mp4"
        svo_path = episode_dir / "recordings" / "SVO" / f"{mp4_id}.svo"

        if not traj_h5.exists():
            raise FileNotFoundError(f"Missing: {traj_h5}")
        if not mp4_path.exists():
            raise FileNotFoundError(f"Missing: {mp4_path}")
        if not svo_path.exists():
            raise FileNotFoundError(f"Missing: {svo_path}")

        return traj_h5, mp4_path, svo_path, episode_dir.name

    def _inject_droid_metadata(
        self,
        metadata: Dict[str, Any],
        traj_h5: Path,
        svo_path: Path,
        mp4_id: str,
    ) -> Dict[str, Any]:
        """
        Inject droid_gt_trace config into metadata so that project_camera_trajectories_to_2d()
        can run the DROID projection path.
        """
        if self.droid_cfg is None:
            raise ValueError("droid_gt_trace config is required to inject DROID metadata")

        view_cam_id = f"{mp4_id}_left"

        droid_meta = {
            "traj_h5": str(traj_h5),
            "svo_path": str(svo_path),
            "view_cam_id": view_cam_id,
            "arm_cam_id": str(self.droid_cfg["arm_cam_id"]),
            "gripper_offset_id": str(self.droid_cfg["gripper_offset_id"]),
            "eye": str(self.droid_cfg.get("eye", "left")),
            "rot_mode": str(self.droid_cfg.get("rot_mode", "euler_xyz")),
        }

        metadata = dict(metadata) if metadata is not None else {}
        metadata["droid_gt_trace"] = droid_meta
        return metadata

    # ------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------
    def process(self, data_dict: Dict[str, Any]):
        """
        Process video and generate camera trajectory visualization.

        DROID (hand=1) return format:
            {
                "head_2d": (T, 2) array,   # optional / usually NaN-filled
                "hand_2d": (T, 2) array,   # single hand(gripper) trajectory in pixel coordinates
                "metadata": {...}
            }
        """
        metadata = data_dict.get("metadata", {}) or {}

        video_frames = data_dict.get("frames", None)
        data_name = data_dict.get("video_name", "unknown")

        # DROID path: If no frames are provided, load from MP4 on disk.
        if video_frames is None and self.droid_cfg is not None:
            mp4_id = data_dict.get("mp4_id", None)
            if mp4_id is None:
                mp4_id = Path(str(data_name)).stem

            ep_dir = data_dict.get("episode_dir", None) or metadata.get("episode_dir", None)
            ep_dir = Path(ep_dir) if ep_dir is not None else None

            traj_h5, mp4_path, svo_path, ep_name = self._resolve_droid_assets(mp4_id=mp4_id, episode_dir=ep_dir)

            video_frames = self._read_mp4_frames_rgb(mp4_path)
            data_name = f"{ep_name}/{mp4_id}"
            metadata = self._inject_droid_metadata(metadata, traj_h5=traj_h5, svo_path=svo_path, mp4_id=mp4_id)

        if video_frames is None:
            raise ValueError("data_dict must contain 'frames' or provide DROID config to load frames from disk")

        viz_height = int(video_frames.shape[1])
        viz_width = int(video_frames.shape[2])

        if self.verbose:
            print(f"[INFO] Processing video: {data_name}")
            print(f"[INFO] Total frames: {len(video_frames)}")
            print(f"[INFO] Video resolution: {viz_width} x {viz_height}")
            print(f"[INFO] Step 1: Projecting 3D trajectories to 2D...")

        trajectory_data = project_camera_trajectories_to_2d(
            metadata=metadata,
            num_frames=len(video_frames),
            viz_width=viz_width,
            viz_height=viz_height,
        )

        if trajectory_data is None:
            print("[WARNING] Failed to project trajectories (missing camera parameters or missing DROID assets)")
            return None

        output_dir = Path(self.output_dir) / data_name
        output_dir.mkdir(parents=True, exist_ok=True)
        traj_output_path = str(output_dir / "robot_trajectory.mp4")

        generate_trajectory_visualization_video(
            video_frames=video_frames,
            trajectory_data=trajectory_data,
            output_path=traj_output_path,
            trace_window=self.trace_window,
        )

        if self.verbose:
            T = trajectory_data["metadata"]["num_frames"]
            print(f"[INFO] Successfully projected {T} frames to 2D pixel coordinates")
            print(f"[INFO]   - head_2d: {trajectory_data['head_2d'].shape}")
            print(f"[INFO]   - hand_2d: {trajectory_data['hand_2d'].shape}")
            print(f"[INFO] Saved: {traj_output_path}")
            print(f"[INFO] Finished processing video {data_name}")


if __name__ == "__main__":
    config = load_config("gt_visual_trace_droid")
    pipeline = GTVisualTraceDroidPipeline(config, verbose=True)

    if "droid_gt_trace" not in config:
        raise KeyError("Config must include 'droid_gt_trace' to run this script in DROID mode")

    base_dir = Path(config["droid_gt_trace"]["base_dir"])
    target_mp4_ids = [str(x) for x in config["droid_gt_trace"]["target_mp4_ids"]]

    traj_files = sorted(base_dir.rglob("trajectory.h5"))
    episode_dirs = sorted({p.parent for p in traj_files})

    for ep_dir in episode_dirs:
        ep_name = ep_dir.name

        for mp4_id in target_mp4_ids:
            mp4_path = ep_dir / "recordings" / "MP4" / f"{mp4_id}.mp4"
            svo_path = ep_dir / "recordings" / "SVO" / f"{mp4_id}.svo"
            traj_h5 = ep_dir / "trajectory.h5"

            if (not mp4_path.exists()) or (not svo_path.exists()) or (not traj_h5.exists()):
                continue

            data_dict = {
                "frames": None,
                "metadata": {"episode_dir": str(ep_dir)},
                "video_name": f"{ep_name}/{mp4_id}.mp4",
                "mp4_id": mp4_id,
                "episode_dir": str(ep_dir),
            }

            pipeline.process(data_dict)

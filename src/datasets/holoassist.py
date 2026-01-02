from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
import tarfile

import cv2
import numpy as np

from .base import BaseDataset

logger = logging.getLogger(__name__)


from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import tarfile

import cv2
import numpy as np

from .base import BaseDataset

SYNC_FPS_POSE = 30.0
SYNC_FPS_HANDS = 30.0
SYNC_FPS_DEPTH = 10.0


class HoloAssistDataset(BaseDataset):
    """
    HoloAssist dataset wrapper.

    Returns:
        Dict containing:
        - frames      : List[np.ndarray], each (Ni, H, W, 3)
        - description : List[str], Action sentences per segment
        - metadata    : {
              "segments": [
                  {"id", "start", "end", "action_sentence", ...}, ...
              ],
              "pose_sync"  : List[np.ndarray], each (Ni_pose, Dp)
              "hands_left" : List[np.ndarray], each (Ni_hl, Dh)
              "hands_right": List[np.ndarray], each (Ni_hr, Dh)
              "depth"      : List[np.ndarray], each (Ni_depth, Hd, Wd)
              "fps"        : float (RGB fps)
              "events"     : full raw events list
              ... extra annotation fields
          }
    """

    def __init__(self, data_dir: str = "vla-dataset-samples/holoassist", load_depth=False):
        self._ann_by_video = self._load_annotations(Path(data_dir))
        self.load_depth = load_depth
        super().__init__(data_dir)

    def get_video_names(self) -> List[str]:
        """
        Discover all video sessions that have a pitch-shifted mp4.
        """
        video_root = Path(self.data_dir) / "video_pitch_shifted"
        if not video_root.exists():
            return []

        names: List[str] = []
        for session_dir in sorted(video_root.iterdir()):
            if not session_dir.is_dir():
                continue
            if (session_dir / "Export_py" / "Video_pitchshift.mp4").exists():
                names.append(session_dir.name)
        return names

    def get_video(self, video_name: str) -> Dict[str, Any]:
        """
        Load a HoloAssist video and return per–'Coarse grained action' segments.
        """
        base = Path(self.data_dir)

        vid_root   = base / "video_pitch_shifted" / video_name / "Export_py"
        hands_root = base / "hands"             / video_name / "Export_py" / "Hands"
        depth_root = base / "ahat_depth"        / video_name / "Export_py"

        video_path      = vid_root / "Video_pitchshift.mp4"
        pose_sync_path  = vid_root / "Video" / "Pose_sync.txt"
        left_sync_path  = hands_root / "Left_sync.txt"
        right_sync_path = hands_root / "Right_sync.txt"
        depth_tar_path  = depth_root / "AhatDepth_synced.tar"

        assert video_path.exists(), f"Missing video at {video_path}"

        frames_full, fps_rgb = self._load_video_frames_and_fps(video_path)  # (N_rgb, H, W, 3)
        pose_full   = self._load_sync_matrix(pose_sync_path)                # (N_pose, Dp)
        hl_full     = self._load_sync_matrix(left_sync_path)                # (N_hl, Dh)
        hr_full     = self._load_sync_matrix(right_sync_path)               # (N_hr, Dh)
        depth_full  = self._load_depth_from_tar(depth_tar_path)             # (N_depth, Hd, Wd)

        ann = self._ann_by_video.get(video_name, None)
        events = ann.get("events", []) if ann is not None else []

        # take only 'Coarse grained action' events
        coarse_events = [e for e in events if e.get("label") == "Coarse grained action"]

        # if no coarse events, fall back to a single segment (whole video)
        if len(coarse_events) == 0:
            descriptions = [(0, len(frames_full)-1, video_name)]
        else:
            descriptions = [(int(np.floor(e["start"] * fps_rgb)), 
                             int(np.ceil(e["end"] * fps_rgb)), 
                             e['attributes']["Action sentence"]) for e in coarse_events]

        # extra annotation-level metadata
        meta_extra: Dict[str, Any] = {}
        if ann is not None:
            meta_extra["batch"] = ann.get("batch")
            meta_extra["taskType"] = ann.get("taskType")
            meta_extra["videoMetadata"] = ann.get("videoMetadata")

        metadata: Dict[str, Any] = {
            "fps": float(fps_rgb),
            "pose_sync": pose_full,
            "hands_left": hl_full,
            "hands_right": hr_full,
            "depth": depth_full,
            "events": events,
        }
        metadata.update(meta_extra)

        return {
            "video_name": video_name,
            "frames": frames_full,          # np.array
            "descriptions": descriptions,    # List[Tuple]
            "metadata": metadata,
        }

    def _load_annotations(self, data_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load data-annotation-trainval-v1_1.json and index by video_name.

        Expected format: list[ {..., "video_name": "...", "events": [...]} ]
        """
        ann_path = data_dir / "data-annotation-trainval-v1_1.json"
        if not ann_path.exists():
            return {}

        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        ann_by_video: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, list):
            for item in data:
                name = item.get("video_name")
                if name is not None:
                    ann_by_video[name] = item
        elif isinstance(data, dict) and "annotations" in data:
            for item in data["annotations"]:
                name = item.get("video_name")
                if name is not None:
                    ann_by_video[name] = item
        return ann_by_video

    def _load_video_frames_and_fps(self, video_path: Path) -> Tuple[np.ndarray, float]:
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) == 0:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8), float(fps or 30.0)
        return np.asarray(frames, dtype=np.uint8), float(fps or 30.0)

    def _load_sync_matrix(self, txt_path: Path) -> np.ndarray:
        """
        Load a *_sync.txt file and return columns 2: as float matrix.
        The first two columns are typically index + absolute time ticks.
        """
        if not txt_path.exists():
            return np.empty((0, 0), dtype=np.float32)
        arr = np.loadtxt(str(txt_path))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, 2:].astype(np.float32)

    def _load_depth_from_tar(self, tar_path: Path) -> np.ndarray:
        """
        Load all PNG depth frames from AhatDepth_synced.tar.
        """
        if not tar_path.exists() and self.load_depth:
            return np.empty((0, 0, 0), dtype=np.uint16)

        depths: List[np.ndarray] = []
        with tarfile.open(tar_path, "r") as tar:
            png_members = sorted(
                [m for m in tar.getmembers() if m.name.lower().endswith(".png")],
                key=lambda m: m.name,
            )
            for m in png_members:
                fobj = tar.extractfile(m)
                if fobj is None:
                    continue
                buf = np.frombuffer(fobj.read(), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                depths.append(img)
        if len(depths) == 0:
            return np.empty((0, 0, 0), dtype=np.uint16)
        return np.asarray(depths)

    @staticmethod
    def _split_stream_by_times(
        arr: np.ndarray,
        segment_times: List[Tuple[float, float]],
        fps: float,
    ) -> List[np.ndarray]:
        """
        Split a time-sampled stream (starting at t=0, sampled at `fps`)
        into segments given [start, end] (seconds).

        Used for:
          - RGB frames (N, H, W, 3)
          - pose_sync (N, Dp)
          - hands_left / hands_right (N, Dh)
          - depth (N, Hd, Wd)

        Returns:
            List[np.ndarray], one segment per (start, end).
            If arr is empty, returns a list of empty slices with the right shape.
        """
        # ensure we have something sliceable
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        num = arr.shape[0] if arr.ndim >= 1 else 0

        # if stream is empty, still return one empty array per segment
        if num == 0:
            empty = arr[0:0]  # shape (0, ...) even if arr was (0,...)
            return [empty for _ in segment_times]

        segs: List[np.ndarray] = []
        for (start_t, end_t) in segment_times:
            start_idx = int(np.floor(start_t * fps))
            end_idx   = int(np.ceil(end_t * fps))

            start_idx = max(start_idx, 0)
            end_idx   = min(end_idx, num)

            if start_idx >= end_idx:
                segs.append(arr[0:0])  # empty segment
            else:
                segs.append(arr[start_idx:end_idx])
        return segs


if __name__ == "__main__":
    dataset = HoloAssistDataset()
    sample = dataset[0]
    print(sample["video_name"])

    frames_segments = sample["frames"]          # List[(Ni, H, W, 3)]
    descs          = sample["descriptions"]      # List[str]

    pose_segments  = sample["metadata"]["pose_sync"]

    for i, (fseg, desc) in enumerate(zip(frames_segments, descs)):
        print(f"\nSegment {i}: {desc}")
        print("  rgb :", fseg.shape)
        print("  pose:", pose_segments[i].shape)

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

import numpy as np
import h5py
import torch
import torchvision

from .base import BaseDataset


# use pyav backend for torchvision VideoReader
torchvision.set_video_backend("pyav")


class AgiBotWorldDataset(BaseDataset):
    def __init__(self, data_dir: str = "vla-dataset-samples/AgiBotWorld-Beta"):
        super().__init__(data_dir)

    def get_video_names(self) -> List[str]:
        data_path = Path(self.data_dir)
        task_info_dir = data_path / "task_info"
        obs_root = data_path / "observations"

        if not task_info_dir.exists() or not obs_root.exists():
            return []

        video_names: List[str] = []

        # task_327.json, task_XXX.json, ...
        for task_file in sorted(task_info_dir.glob("task_*.json")):
            task_id = task_file.stem.split("_")[1]  # "327" from "task_327.json"

            with task_file.open("r") as f:
                episodes = json.load(f)

            for ep in episodes:
                episode_id = str(ep["episode_id"])
                head_path = obs_root / task_id / episode_id / "videos" / "head_color.mp4"
                if head_path.exists():
                    video_names.append(f"{task_id}/{episode_id}")

        return video_names

    def get_video(self, video_name: str) -> Dict[str, Any]:
        """
        Returns per-action splits:

            frames:      List[np.ndarray], each (Ni, H, W, 3) RGB
            description: List[str], each the action_text
            metadata:    dict with:
                - hand_left_frames
                - hand_right_frames
                - action_config
                - task_name
                - init_scene_text
                - camera_params (dict of jsons)
                - proprio_stats (nested dict from h5)
        """
        data_path = Path(self.data_dir)
        task_id, episode_id_str = video_name.split("/")
        episode_id = int(episode_id_str)

        obs_base = data_path / "observations" / task_id / episode_id_str / "videos"
        head_path = obs_base / "head_color.mp4"
        left_path = obs_base / "hand_left_color.mp4"
        right_path = obs_base / "hand_right_color.mp4"

        if not head_path.exists():
            raise FileNotFoundError(f"head_color.mp4 not found at {head_path}")

        head_frames = self._load_video_frames(head_path)          # (T_head, H, W, 3)
        left_frames = self._load_video_frames(left_path) if left_path.exists() else None
        right_frames = self._load_video_frames(right_path) if right_path.exists() else None
        label_info, task_name, init_scene_text = self._load_label_info(
            data_path, task_id, episode_id
        )

        action_cfg = label_info.get("action_config", [])
        descriptions = [(a['start_frame'], a['end_frame'], a["action_text"]) for a in action_cfg]

        camera_params = self._load_camera_params(data_path, task_id, episode_id_str)

        proprio_path = (
            data_path
            / "proprio_stats"
            / task_id
            / episode_id_str
            / "proprio_stats.h5"
        )
        proprio_stats = self._load_hdf5(proprio_path) if proprio_path.exists() else None

        metadata: Dict[str, Any] = {
            "hand_left_frames": left_frames,
            "hand_right_frames": right_frames,
            "action_cfg": action_cfg,
            "task_name": task_name,
            "init_scene_text": init_scene_text,
            "camera_params": camera_params,
            "proprio_stats": proprio_stats,
        }

        return {
            "video_name": video_name,
            "frames": head_frames,           # np.ndarray
            "descriptions": descriptions,     # List[Tuple]
            "metadata": metadata,
        }

    def _load_video_frames(self, video_path: Path) -> np.ndarray:
        """
        Decode full video using torchvision + pyav backend.

        Returns:
            np.ndarray of shape (num_frames, H, W, 3), dtype=uint8
        """
        video_path = str(video_path)

        # ensure backend is pyav
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")

        frames = []
        for frame in reader:
            # frame["data"]: (C, H, W), uint8
            img = frame["data"].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            frames.append(img)

        if len(frames) == 0:
            raise RuntimeError(f"Failed to decode any frames from {video_path}")

        return np.stack(frames, axis=0)

    def _split_by_actions(
        self,
        frames: Optional[np.ndarray],
        action_config: List[Dict[str, Any]],
    ) -> Optional[List[np.ndarray]]:
        """
        Split a (T, ...) array into segments based on action_config
        with fields 'start_frame' and 'end_frame'.

        Returns a list of arrays per action, or None if frames is None.
        """
        if frames is None:
            return None

        T = frames.shape[0]
        segments: List[np.ndarray] = []

        for cfg in action_config:
            s = int(cfg["start_frame"])
            e = int(cfg["end_frame"])
            # clamp to valid range
            s = max(0, min(T, s))
            e = max(s, min(T, e))
            segments.append(frames[s:e])

        return segments

    def _load_label_info(
        self,
        data_path: Path,
        task_id: str,
        episode_id: int,
    ) -> tuple[Dict[str, Any], str, str]:
        """
        Load label_info for a given (task_id, episode_id) from task_info/task_{task_id}.json.
        """
        annot_path = data_path / "task_info" / f"task_{task_id}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annot_path}")

        with annot_path.open("r") as f:
            episodes = json.load(f)

        episode = None
        for ep in episodes:
            if int(ep["episode_id"]) == episode_id:
                episode = ep
                break

        if episode is None:
            raise ValueError(
                f"Episode {episode_id} not found in annotation file {annot_path}"
            )

        label_info = episode.get("label_info", {})
        task_name = episode.get("task_name", "")
        init_scene_text = episode.get("init_scene_text", "")
        return label_info, task_name, init_scene_text

    def _load_camera_params(
        self,
        data_path: Path,
        task_id: str,
        episode_id_str: str,
    ) -> Dict[str, Any]:
        """
        Load all camera param JSONs into a dict[name] = parsed_json.
        """
        cam_root = (
            data_path
            / "parameters"
            / task_id
            / episode_id_str
            / "parameters"
            / "camera"
        )

        camera_params: Dict[str, Any] = {}
        if not cam_root.exists():
            return camera_params

        # 1) Load all raw JSONs under filename keys
        for jf in sorted(cam_root.glob("*.json")):
            fname = jf.name
            if fname.endswith("_extrinsic_params_aligned.json"):
                with jf.open("r") as f:
                    content = json.load(f)
                    prefix = fname.replace("_extrinsic_params_aligned.json", "")

                extr_list = []

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "extrinsic" in item:
                            extr_list.append(item["extrinsic"])
                        else:
                            extr_list.append(item)
                elif isinstance(content, dict):
                    if "extrinsic" in content:
                        extr_list.append(content["extrinsic"])
                    else:
                        extr_list.append(content)
                else:
                    # unexpected type, just wrap as-is
                    extr_list.append(content)

                camera_params[f"{prefix}_extrinsics"] = extr_list

            elif fname.endswith("_intrinsic_params.json"):
                with jf.open("r") as f:
                    content = json.load(f)
                    prefix = fname.replace("_extrinsic_params_aligned.json", "")

                intr = None

                if isinstance(content, list):
                    # take first element
                    first = content[0] if len(content) > 0 else None
                    if isinstance(first, dict) and "intrinsic" in first:
                        intr = first["intrinsic"]
                    else:
                        intr = first
                elif isinstance(content, dict):
                    intr = content.get("intrinsic", content)
                else:
                    intr = content

                if intr is not None:
                    camera_params[f"{prefix}_intrinsics"] = intr

        return camera_params

    def _load_hdf5(self, hdf5_path: Path) -> Dict[str, Any]:
        """
        Recursively load an HDF5 file into nested dicts of numpy arrays.
        """
        def _recursive_load(group):
            result = {}
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    result[key] = _recursive_load(item)
                else:
                    result[key] = item[()]
            return result

        with h5py.File(hdf5_path, "r") as f:
            return _recursive_load(f)


if __name__ == "__main__":
    ds = AgiBotWorldDataset()
    print(f"data_dir: {ds.data_dir}")
    print(f"#videos: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print("video_name:", sample["video_name"])
        print("#actions:", len(sample["frames"]))
        print("frames[0].shape:", sample["frames"][0].shape if sample["frames"] else None)
        print("descriptions[0]:", sample["descriptions"][0] if sample["descriptions"] else None)
        print("metadata keys:", sample["metadata"].keys())

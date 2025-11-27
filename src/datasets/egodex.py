from typing import List, Dict, Any
from pathlib import Path

import cv2
import h5py
import numpy as np

from .base import BaseDataset


class EgoDexDataset(BaseDataset):
    def __init__(self, data_dir: str = "vla-dataset-samples/egodex"):
        super().__init__(data_dir)

    def get_video_names(self) -> List[str]:
        data_path = Path(self.data_dir)
        if not data_path.exists():
            return []

        video_files = [
            f for f in data_path.rglob("*.mp4")
            if "_resized" not in f.name
        ]

        return [
            f"{f.parent.parent.name}/{f.parent.name}/{f.stem}"
            for f in sorted(video_files)
        ]

    def get_video(self, video_name: str) -> Dict[str, Any]:
        """Load a single video from the EgoDex dataset.

        Args:
            video_name: Video identifier in format "{part_name}/{task_name}/{video_id}"

        Returns:
            Dict containing:
                - video_name (str): The video identifier
                - frames (np.ndarray): Shape (num_frames, 1080, 1920, 3) RGB frames
                - description (str): Task name as natural language
                - metadata (dict): Camera params, hand keypoints (MANO), transforms, etc.
        """
        parts = video_name.split("/")
        part_name, task_name, video_id = parts[0], parts[1], parts[2]

        base_path = Path(self.data_dir) / part_name / task_name / video_id
        video_path = base_path.with_suffix(".mp4")
        hdf5_path = base_path.with_suffix(".hdf5")
        mano_path = base_path.parent / f"{video_id}_mano.hdf5"

        assert video_path.exists(), f"Video file not found: {video_path}"
        frames = self._load_video_frames(video_path)
        description = task_name.replace("_", " ")

        metadata = {}
        if hdf5_path.exists():
            metadata.update(self._load_hdf5(hdf5_path))
        if mano_path.exists():
            metadata["mano"] = self._load_hdf5(mano_path)

        return {
            "video_name": video_name,
            "frames": frames,
            "description": description,
            "metadata": metadata,
        }

    def _load_video_frames(self, video_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.array(frames)

    def _load_hdf5(self, hdf5_path: Path) -> Dict[str, Any]:
        def _recursive_load(group):
            result = {}
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Group):
                    result[key] = _recursive_load(item)
                else:
                    result[key] = item[:]
            return result

        with h5py.File(hdf5_path, "r") as f:
            return _recursive_load(f)


if __name__ == "__main__":
    dataset = EgoDexDataset()
    print(f"Data directory: {dataset.data_dir}")
    print(f"Number of videos: {len(dataset)}")
    print(f"Video names: {dataset.video_names}\n")

    data_dict = dataset[0]
    print(f"\nvideo_name: {data_dict['video_name']}")
    print(f"frames shape: {data_dict['frames'].shape}")
    print(f"description: {data_dict['description']}")
    print(f"\nmetadata keys: {list(data_dict['metadata'].keys())}")

    for key, value in data_dict['metadata'].items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    print(f"  {subkey}: {subvalue.shape}")
                else:
                    print(f"  {subkey}: {subvalue}")
        elif isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
import logging
from pathlib import Path
from typing import Any, Dict, List


class BaseDataset:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.video_names = self.get_video_names()
        logging.info(f"Loaded {self.__class__.__name__} with {len(self.video_names)} videos")

    def get_video_names(self) -> List[str]:
        raise NotImplementedError

    def get_video(self, video_name: str) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.video_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_name = self.video_names[idx]
        return self.get_video(video_name)

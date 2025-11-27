import logging
from pathlib import Path
from typing import Any, Dict, List


class BaseDataset:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.episode_names = self.get_episode_names()
        logging.info(f"Loaded {self.__class__.__name__} with {len(self.episode_names)} episodes")

    def get_episode_names(self) -> List[str]:
        raise NotImplementedError

    def get_episode(self, episode_name: str) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.episode_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_name = self.episode_names[idx]
        return self.get_episode(episode_name)

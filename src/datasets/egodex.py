

from .base import BaseDataset


class EgodexDataset(BaseDataset):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)

    def get_video_names(self) -> List[str]:
        return [f"video_{i}.mp4" for i in range(len(self.video_names))]
import logging
from pathlib import Path
from typing import Any, Dict, List


class BaseDataset:
    """Base class for video datasets.

    Subclasses should implement:
        - get_video_names(): Returns list of video identifiers
        - get_video(video_name): Returns dict with video data

    The get_video() method must return a dict containing:
        - video_name (str): The video identifier
        - frames (np.ndarray): Video frames of shape (num_frames, H, W, 3)
        - description (str): Natural language description of the video
        - metadata (dict): Additional dataset-specific metadata
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.video_names = self.get_video_names()
        logging.info(f"Loaded {self.__class__.__name__} with {len(self.video_names)} videos")

    def get_video_names(self) -> List[str]:
        """Return list of video identifiers.

        Returns:
            List of string identifiers for all videos in the dataset.
        """
        raise NotImplementedError

    def get_video(self, video_name: str) -> Dict[str, Any]:
        """Load and return data for a specific video.

        Args:
            video_name: Video identifier from get_video_names()

        Returns:
            Dict containing:
                - video_name (str): The video identifier
                - frames (np.ndarray): Shape (num_frames, H, W, 3) RGB frames
                - description (str): Natural language description
                - metadata (dict): Additional dataset-specific data
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.video_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_name = self.video_names[idx]
        print(f"Loading video: {video_name}")
        return self.get_video(video_name)

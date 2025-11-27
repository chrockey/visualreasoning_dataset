from typing import Any, Dict

from .base import BasePipeline, load_config
from ..models.gemma import Gemma

# TODO : Implement the VisualTracePipeline
# 1. Load the Gemma, CoTracker v3, Grounded-SAM2 models
# 2. extract main object from the image using Gemma
# 3. Grounded SAM2 to segment the main object and extract the keypoints
# 4. track the keypoints using CoTracker v3
# 5. filter the keypoints using the confidence score

class VisualTracePipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.object_extractor = Gemma(config["gemma"]["model_id"])
        self.grounded_segmenter = None
        self.keypoint_tracker = None
        
    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        raise NotImplementedError


if __name__ == "__main__":
    config = load_config("visual_trace")
    pipeline = VisualTracePipeline(config)

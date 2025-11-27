from typing import Any, Dict

from .base import BasePipeline, load_config


class VisualTracePipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        raise NotImplementedError


if __name__ == "__main__":
    config = load_config("visual_trace")
    pipeline = VisualTracePipeline(config)

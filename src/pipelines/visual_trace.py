from typing import Any, Dict

from .base import BasePipeline


class VisualTracePipeline(BasePipeline):
    def __init__(self, args, kwargs):
        super().__init__()

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        raise NotImplementedError


if __name__ == "__main__":
    pipeline = VisualTracePipeline()

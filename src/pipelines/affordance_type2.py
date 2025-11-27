from typing import Any, Dict

from src.models.molmo import Molmo
from src.models.sam2 import SAM2

from .base import BasePipeline


class AffordanceType2Pipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.molmo = Molmo()
        self.sam2 = SAM2()

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        raise NotImplementedError


if __name__ == "__main__":
    pipeline = AffordanceType2Pipeline()

from typing import Any, Dict

from src.models.molmo import Molmo
from src.models.sam2 import SAM2

from .base import BasePipeline, load_config


class AffordanceType2Pipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.molmo = Molmo(**config.get("molmo", {}))
        self.sam2 = SAM2(**config.get("sam2", {}))

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        raise NotImplementedError


if __name__ == "__main__":
    config = load_config("affordance_type2")
    pipeline = AffordanceType2Pipeline(config)

import logging
from typing import Any, Dict


class BasePipeline:
    def __init__(self):
        logging.info(f"Loaded {self.__class__.__name__}")

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, data_dict: Dict[str, Any], save_dir: str):
        data_dict = self.preprocess(data_dict)
        results = self.process(data_dict)
        return results

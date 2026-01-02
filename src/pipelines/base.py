import logging
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def load_config(name: str) -> Dict[str, Any]:
    """Load config from config/{name}.yaml"""
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


class BasePipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logging.info(f"Loaded {self.__class__.__name__}")

    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, data_dict: Dict[str, Any], save_dir: str):
        data_dict = self.preprocess(data_dict)
        results = self.process(data_dict)
        return results
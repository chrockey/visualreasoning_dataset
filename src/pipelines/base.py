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

def load_dataset_from_config(config: Dict[str, Any]):
    """Load dataset instance from config['dataset'] specification."""
    from src.datasets import DATASETS

    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name")

    if not dataset_name:
        raise ValueError(
            "No dataset specified in config. Please set 'dataset.name' in config file.\n"
            f"Available datasets: {list(DATASETS.keys())}"
        )

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {list(DATASETS.keys())}"
        )

    # Instantiate dataset
    dataset_cls = DATASETS[dataset_name]
    dataset_dir = dataset_config.get("dir")

    if dataset_dir:
        dataset = dataset_cls(data_dir=dataset_dir)
    else:
        dataset = dataset_cls()

    if len(dataset) == 0:
        raise RuntimeError(
            f"No data found for dataset '{dataset_name}'. "
            f"Please check the data directory: {dataset.data_dir}"
        )

    logging.info(f"Loaded {dataset_name} dataset with {len(dataset)} samples")
    return dataset


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

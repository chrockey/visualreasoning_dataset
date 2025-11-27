"""
Pipeline Worker - Runs pipelines on jobs from the job server.

Usage:
    python -m src.job_server.pipeline_worker --config config.yaml

Config file (YAML):
    server:
        url: http://localhost:8000
        poll_interval: 5.0

    experiment:
        id: exp-001  # Required: experiment to fetch jobs from

    worker:
        id: worker-1  # optional, auto-generated if not set
        max_jobs: null  # null = unlimited
        save_dir: /path/to/results
        stop_when_empty: true  # Stop when no more jobs

    pipeline:
        name: affordance_type1
        # Pipeline-specific arguments passed to __init__
        model_size: large
        threshold: 0.5
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.job_server.worker import BaseWorker

logger = logging.getLogger(__name__)

# Registry of available pipelines
PIPELINES = {
    "affordance_type1": "src.pipelines.affordance_type1.AffordanceType1Pipeline",
    "affordance_type2": "src.pipelines.affordance_type2.AffordanceType2Pipeline",
    "visual_trace": "src.pipelines.visual_trace.VisualTracePipeline",
}


def load_pipeline(pipeline_name: str, config: Optional[Dict[str, Any]] = None):
    """
    Dynamically load and instantiate a pipeline class.

    Args:
        pipeline_name: Name of the pipeline (key in PIPELINES registry)
        config: Optional dict of kwargs to pass to pipeline __init__
    """
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(PIPELINES.keys())}")

    module_path, class_name = PIPELINES[pipeline_name].rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    pipeline_class = getattr(module, class_name)

    config = config or {}
    return pipeline_class(**config)


class PipelineWorker(BaseWorker):
    """Worker that runs a pipeline on jobs."""

    def __init__(
        self,
        server_url: str,
        experiment_id: str,
        pipeline_name: str,
        save_dir: str,
        pipeline_config: Optional[Dict[str, Any]] = None,
        worker_id: str = None,
        **kwargs,
    ):
        super().__init__(server_url, experiment_id=experiment_id, worker_id=worker_id, **kwargs)
        self.pipeline_name = pipeline_name
        self.pipeline_config = pipeline_config or {}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None

    def setup(self):
        """Load the pipeline (and its models like Molmo, SAM2)."""
        logger.info(f"Loading pipeline: {self.pipeline_name}")
        if self.pipeline_config:
            logger.info(f"Pipeline config: {self.pipeline_config}")
        self.pipeline = load_pipeline(self.pipeline_name, self.pipeline_config)
        logger.info("Pipeline loaded successfully")

    def process_job(self, job_id: str, payload: Dict[str, Any]):
        """
        Process a single job through the pipeline.

        Expected payload format:
            {
                "video_path": "/path/to/video.mp4",
                # ... other pipeline-specific data
            }
        """
        # Create job-specific save directory
        job_save_dir = self.save_dir / job_id
        job_save_dir.mkdir(parents=True, exist_ok=True)

        # Run the pipeline
        results = self.pipeline(payload, save_dir=str(job_save_dir))

        logger.info(f"Job {job_id} results saved to {job_save_dir}")
        return results

    def teardown(self):
        """Clean up resources."""
        if self.pipeline is not None:
            # Free GPU memory if pipeline has cleanup method
            if hasattr(self.pipeline, "cleanup"):
                self.pipeline.cleanup()
            self.pipeline = None


def main():
    parser = argparse.ArgumentParser(description="Run pipeline worker")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    server_config = config.get("server", {})
    experiment_config = config.get("experiment", {})
    worker_config = config.get("worker", {})
    pipeline_config = config.get("pipeline", {})

    # Extract pipeline name, rest goes to pipeline __init__
    pipeline_name = pipeline_config.pop("name")

    # Experiment ID is required
    experiment_id = experiment_config.get("id")
    if not experiment_id:
        raise ValueError("experiment.id is required in config")

    worker = PipelineWorker(
        server_url=server_config.get("url", "http://localhost:8000"),
        experiment_id=experiment_id,
        pipeline_name=pipeline_name,
        save_dir=worker_config["save_dir"],
        pipeline_config=pipeline_config,
        worker_id=worker_config.get("id"),
        poll_interval=server_config.get("poll_interval", 5.0),
        stop_when_empty=worker_config.get("stop_when_empty", False),
    )
    worker.run(max_jobs=worker_config.get("max_jobs"))


if __name__ == "__main__":
    main()

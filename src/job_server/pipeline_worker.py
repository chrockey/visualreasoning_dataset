"""
Pipeline Worker - Runs pipelines on jobs from the job server.

Usage:
    python -m src.job_server.pipeline_worker --experiment EXP_ID [--server URL]

Config structure:
    config/{experiment_config}.yaml  - Worker settings + pipeline name
        worker:
            save_dir: /path/to/results
            stop_when_empty: true
        pipeline:
            name: gt_visual_trace_droid  # or sam3_tracker, gemini_sam3_tracker, etc.

    config/{pipeline_name}.yaml  - Pipeline-specific settings
        sam3:
            model_id: facebook/sam3
            gpus_to_use: [0]
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
    # GT Visual Trace pipelines (dataset-specific)
    "gt_visual_trace_droid": "src.pipelines.gt_visual_trace.droid.GTVisualTraceDroidPipeline",
    "gt_visual_trace_agibotworld": "src.pipelines.gt_visual_trace.agibotworld.GTVisualTraceAgiBotWorldPipeline",
    # Visual Trace pipelines (dataset-agnostic)
    "sam3_tracker": "src.pipelines.visual_trace.sam3_tracker.SAM3TrackerPipeline",
    "gemini_sam3_tracker": "src.pipelines.visual_trace.gemini_sam3_tracker.GeminiSAM3TrackerPipeline",
}


def load_pipeline(pipeline_name: str, config: Dict[str, Any]):
    """
    Dynamically load and instantiate a pipeline class.

    Args:
        pipeline_name: Name of the pipeline (key in PIPELINES registry)
        config: Config dict to pass to pipeline __init__
    """
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(PIPELINES.keys())}")

    module_path, class_name = PIPELINES[pipeline_name].rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    pipeline_class = getattr(module, class_name)

    return pipeline_class(config)


class PipelineWorker(BaseWorker):
    """Worker that runs a pipeline on jobs."""

    def __init__(
        self,
        server_url: str,
        experiment_id: str,
        pipeline_name: str,
        pipeline_config: Dict[str, Any],
        save_dir: str,
        worker_id: str = None,
        **kwargs,
    ):
        super().__init__(server_url, experiment_id=experiment_id, worker_id=worker_id, **kwargs)
        self.pipeline_name = pipeline_name
        self.pipeline_config = pipeline_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None

    def setup(self):
        """Load the pipeline (and its models like Molmo, SAM2)."""
        logger.info(f"Loading pipeline: {self.pipeline_name}")
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
    import requests

    parser = argparse.ArgumentParser(description="Run pipeline worker")
    parser.add_argument("--experiment", required=True, help="Experiment ID to process")
    parser.add_argument("--server", default="http://localhost:8000", help="Job server URL")
    parser.add_argument("--worker-id", help="Worker ID (auto-generated if not set)")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Poll interval in seconds")
    parser.add_argument("--max-jobs", type=int, help="Maximum jobs to process")
    args = parser.parse_args()

    from src.pipelines.base import load_config

    # Fetch experiment info to get config file path
    resp = requests.get(f"{args.server}/experiments/{args.experiment}")
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch experiment: {resp.text}")

    experiment = resp.json()
    config_file = experiment.get("config_file")
    if not config_file:
        raise ValueError(f"Experiment {args.experiment} has no config_file set")

    # Load experiment config (worker settings + pipeline name)
    config_path = Path(__file__).parent.parent.parent / "config" / config_file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    worker_config = config.get("worker", {})
    pipeline_name = config.get("pipeline", {}).get("name")
    if not pipeline_name:
        raise ValueError(f"No pipeline.name specified in {config_file}")

    # Load pipeline config from config/{pipeline_name}.yaml
    pipeline_config = load_config(pipeline_name)

    worker = PipelineWorker(
        server_url=args.server,
        experiment_id=args.experiment,
        pipeline_name=pipeline_name,
        pipeline_config=pipeline_config,
        save_dir=worker_config["save_dir"],
        worker_id=args.worker_id or worker_config.get("id"),
        poll_interval=args.poll_interval,
        stop_when_empty=worker_config.get("stop_when_empty", False),
    )
    worker.run(max_jobs=args.max_jobs or worker_config.get("max_jobs"))


if __name__ == "__main__":
    main()

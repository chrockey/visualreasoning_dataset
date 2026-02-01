# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Distributed pipeline system for visual reasoning dataset annotation using SAM3 (tracking) and Gemini (refinement).

## Commands

### Installation
```bash
./install.sh
```

### Run Job Server
```bash
uvicorn src.job_server.server:app --host 0.0.0.0 --port 8000
```
Admin panel available at http://localhost:8000/admin

### Run Pipeline Worker
```bash
python -m src.job_server.pipeline_worker --experiment EXPERIMENT_ID [--server URL]
```
The worker fetches experiment info to get the config file path, then loads pipeline settings from that file.

### Test Datasets
```bash
python -m src.datasets.oxe
python -m src.datasets.agibotworld
```

### Test Individual Pipeline
```bash
# GT Visual Trace pipelines (dataset-specific)
python -m src.pipelines.gt_visual_trace.droid
python -m src.pipelines.gt_visual_trace.agibotworld

# Visual Trace pipelines (dataset-agnostic, requires --config)
python -m src.pipelines.visual_trace.sam3_tracker --config visual_trace_language_table --data-dir ./data --episode-index 0
python -m src.pipelines.visual_trace.gemini_sam3_tracker --config visual_trace_bridge --data-dir ./data --episode-index 0
```

### Client CLI
```bash
python -m src.job_server.client create --name "My Experiment"
python -m src.job_server.client list
python -m src.job_server.client get EXPERIMENT_ID
python -m src.job_server.client delete EXPERIMENT_ID
```

## Architecture

### Datasets (`src/datasets/`)
All datasets extend `BaseDataset` and provide:
- `video_name` (str): Video/episode identifier for tracking annotations
- `frames` (np.ndarray): Video frames as (N, H, W, 3) RGB array
- `description` (str): Natural language task description
- `metadata` (dict): Dataset-specific annotations

**AgiBotWorld** (`agibotworld.py`): Bimanual manipulation dataset
- Data: MP4 videos + JSON metadata (camera params)
- Used by: GT Visual Trace pipeline

**Open X-Embodiment** (`oxe.py`): Robot manipulation episodes from TFRecord shards
- Datasets: DROID, Language Table, Bridge
- Data: TFRecord files with robot states, actions, language instructions
- Used by: GT Visual Trace (DROID), SAM3 Tracker (Language Table), Gemini+SAM3 (Bridge)

### Models (`src/models/`)
- **SAM3VideoTracker**: Video object tracking and segmentation using SAM3. Supports text-based prompting and mask propagation across video frames.

### Pipelines (`src/pipelines/`)
All pipelines extend `BasePipeline` and implement:
- `preprocess(data_dict)` - prepare input data
- `process(data_dict)` - run the pipeline logic
- `__call__(data_dict, save_dir)` - orchestrates preprocess → process

Register new pipelines in `PIPELINES` dict in `src/job_server/pipeline_worker.py`.

### Job Server (`src/job_server/`)
FastAPI + SQLAlchemy + SQLite job distribution system with experiments containing jobs. Includes SQLAdmin panel at `/admin`.

- **server.py**: REST API with endpoints for experiments and jobs. Uses SQLAlchemy ORM. Jobs have states: `pending` → `running` → `completed`/`failed`.
- **worker.py**: `BaseWorker` abstract class - subclass and implement `setup()` and `process_job(job_id, payload)`.
- **pipeline_worker.py**: `PipelineWorker` loads a pipeline by name and runs it on jobs. Configured via YAML.
- **client.py**: `JobClient` for submitting jobs and monitoring. Also has CLI interface.

### Job Flow
1. Create experiment via client with `config_file` path
2. Submit jobs (each has job_id + payload dict)
3. Workers fetch experiment info to get config, then process jobs
4. Failed jobs auto-retry up to `MAX_JOB_RETRIES` (default 3) times
5. Experiment status auto-updates to `completed` when all jobs are done

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Distributed pipeline system for visual reasoning dataset annotation using Molmo (VLM for point extraction) and SAM2 (segmentation).

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
python -m src.datasets.egodex
python -m src.datasets.oxe
```

### Test Individual Pipeline
```bash
python -m src.pipelines.affordance_type1
python -m src.pipelines.affordance_type2
python -m src.pipelines.visual_trace
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

**EgoDex** (`egodex.py`): Egocentric hand manipulation videos
- Format: `video_name = "{part}/{task}/{video_id}"` (e.g., `"part1/add_remove_lid/0"`)
- Data: MP4 videos + HDF5 metadata (camera params, MANO hand poses, joint transforms)
- Frames: (N, 1080, 1920, 3)

**Open X-Embodiment** (`oxe.py`): Robot manipulation episodes from TFRecord shards
- Format: `video_name = "{dataset}/{shard_id}/{episode_in_shard}"` (e.g., `"asu_table_top.../00000/0"`)
- Data: TFRecord files with robot states, actions, language instructions, embeddings
- Frames: (N, 224, 224, 3)
- Metadata includes `tfrecord_info` dict for shard-based annotation tracking:
  - `dataset_name`, `shard_idx`, `episode_in_shard`, `split`, `tfrecord_path`
- Annotations should be saved by shard: group episodes by `{dataset}/{shard_id}` for input-annotation matching

### Models (`src/models/`)
- **Molmo**: VLM wrapper that extracts (x, y) point coordinates from images via natural language queries. Uses `extract_points()` to parse model output into pixel coordinates.
- **SAM2**: Segmentation model that takes images + point coordinates and produces masks. Supports three mask selection modes: `highest_score`, `smallest_mask`, `random`.

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

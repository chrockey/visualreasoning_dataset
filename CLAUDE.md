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

### Run Pipeline Worker
```bash
cp config/config.example.yaml config.yaml
# Edit config.yaml
python -m src.job_server.pipeline_worker --config config.yaml
```

### Test Individual Pipeline
```bash
python -m src.pipelines.affordance_type1
python -m src.pipelines.affordance_type2
python -m src.pipelines.visual_trace
```

### Client CLI
```bash
python -m src.job_server.client create-experiment exp-001 --name "My Experiment"
python -m src.job_server.client submit exp-001 job-001 '{"video_path": "/data/video.mp4"}'
python -m src.job_server.client stats exp-001
python -m src.job_server.client wait exp-001
```

## Architecture

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
FastAPI + SQLite job distribution system with experiments containing jobs.

- **server.py**: REST API with endpoints for experiments and jobs. Jobs have states: `pending` → `running` → `completed`/`failed`.
- **worker.py**: `BaseWorker` abstract class - subclass and implement `setup()` and `process_job(job_id, payload)`.
- **pipeline_worker.py**: `PipelineWorker` loads a pipeline by name and runs it on jobs. Configured via YAML.
- **client.py**: `JobClient` for submitting jobs and monitoring. Also has CLI interface.

### Job Flow
1. Create experiment via client
2. Submit jobs (each has job_id + payload dict)
3. Workers fetch pending jobs, mark as running, process, mark as completed/failed
4. Monitor via `client.stats()` or `client.wait_for_completion()`

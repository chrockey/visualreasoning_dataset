# Visual Reasoning Annotation

Distributed pipeline system for visual reasoning dataset annotation using Molmo and SAM2.

## Installation

```bash
./install.sh
```

Or manually:

```bash
pip install torch torchvision
pip install transformers
pip install --no-deps --no-build-isolation git+https://github.com/facebookresearch/sam2.git
pip install --no-deps --no-build-isolation flash-attn
pip install fastapi uvicorn pyyaml requests
```

## Project Structure

```
├── config/
│   └── config.example.yaml    # Worker config template
├── src/
│   ├── models/                # Model wrappers
│   │   ├── molmo.py
│   │   └── sam2.py
│   ├── pipelines/             # <-- WORK HERE: Add your pipelines
│   │   ├── base.py
│   │   ├── affordance_type1.py
│   │   ├── affordance_type2.py
│   │   └── visual_trace.py
│   └── job_server/            # Distributed job system
│       ├── server.py
│       ├── worker.py
│       ├── client.py
│       └── pipeline_worker.py
└── install.sh
```

## Pipelines

Each pipeline processes video/image data for annotation. Work on pipelines in `src/pipelines/`.

### Testing a Pipeline

Each pipeline has a `if __name__ == "__main__"` block for standalone testing:

```bash
# Test individual pipelines
python -m src.pipelines.affordance_type1
python -m src.pipelines.affordance_type2
python -m src.pipelines.visual_trace
```

### Creating a New Pipeline

```python
# src/pipelines/my_pipeline.py
from typing import Any, Dict
from src.models.molmo import Molmo
from src.models.sam2 import SAM2
from .base import BasePipeline


class MyPipeline(BasePipeline):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.molmo = Molmo()
        self.sam2 = SAM2()

    def preprocess(self, data_dict: Dict[str, Any]):
        return data_dict

    def process(self, data_dict: Dict[str, Any]):
        return {"result": "..."}


if __name__ == "__main__":
    # Test with sample data
    pipeline = MyPipeline(threshold=0.7)
    sample = {"video_path": "/path/to/test.mp4"}
    result = pipeline(sample, save_dir="/tmp/test")
    print(result)
```

Then register in `src/job_server/pipeline_worker.py`:

```python
PIPELINES = {
    ...
    "my_pipeline": "src.pipelines.my_pipeline.MyPipeline",
}
```

## Distributed Processing

For processing large datasets across multiple machines/GPUs.

### 1. Start Server

```bash
uvicorn src.job_server.server:app --host 0.0.0.0 --port 8000
```

### 2. Submit Jobs

```python
from src.job_server.client import JobClient

client = JobClient("http://localhost:8000")

# Submit jobs
jobs = [(f"video-{i}", {"video_path": f"/data/video_{i}.mp4"}) for i in range(1000)]
client.submit_batch(jobs)
```

### 3. Run Workers

```bash
# Copy config template
cp config/config.example.yaml config.yaml
# Edit config.yaml with your settings

# Run worker
python -m src.job_server.pipeline_worker --config config.yaml
```

### 4. Monitor Progress

```bash
python -m src.job_server.client stats --server http://localhost:8000
```

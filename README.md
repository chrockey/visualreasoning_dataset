# Visual Reasoning Annotation

Pipeline system for visual reasoning dataset annotation using Molmo and SAM2.

## Installation

```bash
./install.sh
```

## Supported Datasets

| Dataset | Status | Additional Metadata |
|---------|--------|---------------------|
| **EgoDex** | ✅ | Camera intrinsics, joint transforms (70+ joints), confidence scores, MANO hand poses |
| **AgiBotWorld** | ⬜️ | TBD |
| **HoloAssist** | ⬜️ | TBD |
| **Open-X-Embodiment** | ⬜️ | TBD |

All datasets provide:
- `frames`: Video frames as (N, H, W, 3) numpy array
- `description`: Task description string
- `metadata`: Dataset-specific annotations and additional data

## Project Structure

```
src/
├── models/                # Model wrappers
│   ├── molmo.py          # VLM for point extraction
│   └── sam2.py           # Segmentation model
└── pipelines/            # <-- WORK HERE
    ├── base.py
    ├── affordance_type1.py
    ├── affordance_type2.py
    └── visual_trace.py
```

## Testing Pipelines

```bash
python -m src.pipelines.affordance_type1
python -m src.pipelines.affordance_type2
python -m src.pipelines.visual_trace
```

## Creating a New Pipeline

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
    pipeline = MyPipeline(threshold=0.7)
    result = pipeline({"video_path": "/path/to/test.mp4"}, save_dir="/tmp/test")
    print(result)
```

Then register in `src/job_server/pipeline_worker.py`:

```python
PIPELINES = {
    ...
    "my_pipeline": "src.pipelines.my_pipeline.MyPipeline",
}
```

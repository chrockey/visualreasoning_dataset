# Visual Reasoning Annotation

Pipeline system for visual reasoning dataset annotation using Molmo and SAM2.

## Installation

```bash
./install.sh
```

## Supported Datasets

| Dataset | Status | Additional Metadata |
|---------|--------|---------------------|
| **EgoDex** | ✅ | Camera parameters (intrinsics/extrinsics), joint transforms (70+ joints), confidence scores, MANO hand poses |
| **Open X-Embodiment** | ✅ | Robot states (joint angles), actions, 512-dim language embeddings |
| **AgiBotWorld** | ✅ | Camera parameters (intrinsics/extrinsics), additional views, actions, proprio_stats |
| **HoloAssist** | ✅ | Hand poses (left/right), depth |

All datasets inherit from `BaseDataset` and provide:
- `video_name`: Video identifier string (for tracking annotations)
- `frames`: Video frames as (N, H, W, 3) numpy array
- `description`: Task description string
- `metadata`: Dataset-specific annotations and additional data

### Using Datasets

```python
from src.datasets.egodex import EgoDexDataset
from src.datasets.oxe import OXEDataset
from src.datasets.agibotworld import AgiBotWorldDataset
from src.datasets.holoassist import HoloAssistDataset

# EgoDex: Egocentric hand manipulation videos
dataset = EgoDexDataset()  # Default: vla-dataset-samples/egodex
print(f"Videos: {len(dataset)}")
data = dataset[0]
# data['video_name']: "part1/add_remove_lid/0"
# data['frames']: (288, 1080, 1920, 3)
# data['metadata']: camera, MANO hand poses, transforms

# Open X-Embodiment: Robot manipulation episodes
dataset = OXEDataset()  # Default: vla-dataset-samples/open-x-embodiment
print(f"Episodes: {len(dataset)}")
data = dataset[0]
# data['video_name']: "asu_table_top_converted_externally_to_rlds/00000/0"
#                     {dataset_name}/{shard_id}/{episode_in_shard}
# data['frames']: (355, 224, 224, 3)
# data['metadata']['state']: robot joint states
# data['metadata']['action']: robot actions
# data['metadata']['language_embedding']: 512-dim embedding
# data['metadata']['tfrecord_info']: shard tracking info for annotations

# AgiBotWorld-Beta: Bimanual manipulation dataset
dataset = AgiBotWorldDataset()
print(f"Episodes: {len(dataset)}")
data = dataset[0]
# [f.shape for f in data['frames']]
# [(151, 480, 640, 3),
#  (239, 480, 640, 3),
#  (165, 480, 640, 3),
#  (197, 480, 640, 3),
#  (168, 480, 640, 3),
#  (276, 480, 640, 3)]
#  data['description']
# ['Retrieve cucumber from the shelf.',
#  'Place the held cucumber into the plastic bag in the shopping cart.',
#  'Retrieve tomato from the shelf.',
#  'Place the held tomato into the plastic bag in the shopping cart.',
#  'Retrieve corn from the shelf.',
#  "Place the held corn into the shopping cart's plastic bag."]
# data['metadata']['hand_left_frames]: frames from hand-left cam
# data['metadata']['hand_right_frames]: frames from hand-right cam
# data['metadata']['action_config']: action text, skill(pick,place,..)
# data['metadata']['proprio_stats]: effector (orientation, velocity, ..)

# HoloAssist: Egocentric human interaction dataset
dataset = HoloAssistDataset()
print(f"Videos: {len(dataset)}")
data = dataset[0]
# [f.shape for f in data['frames']]
# [(423, 504, 896, 3),
#  (889, 504, 896, 3),
#  (5772, 504, 896, 3),
#  (171, 504, 896, 3),
#  (272, 504, 896, 3),
#  (616, 504, 896, 3),
#  (185, 504, 896, 3),
#  (643, 504, 896, 3),
#  (353, 504, 896, 3)]
# data['description']
# ['The student grabs the GoPro.',
#  'The student changes the battery for the GoPro.',
#  'The student opens the GoPro.',
#  'The student turns on their GoPro.',
#  'The student turns off the gopro.',
#  'The student assembles the mounting_peg.',
#  'The student disassemble the mounting_peg.',
#  'The student assemble handheld_grip.',
#  'The students disassemble the handheld_grip.']
# data['metadata']['depth']: Depth
# data['metadata']['hands_left']: Hand pose (left)
# data['metadata']['hands_right']: Hand pose (right)
# data['metadata']['pose_sync']: Camera pose

```

## Project Structure

```
src/
├── datasets/            # Dataset loaders
│   ├── base.py          # BaseDataset abstract class
│   ├── agibotworld.py   # AgiBotWorld-Beta (Bimanual manipulation robot manipulation)
│   ├── egodex.py        # EgoDex (egocentric hand manipulation)
│   ├── holoassist.py    # HoloAssist (Egocentric human interaction)
│   └── oxe.py           # Open X-Embodiment (robot manipulation)
├── models/              # Model wrappers
│   ├── molmo.py         # VLM for point extraction
│   └── sam2.py          # Segmentation model
├── pipelines/           # <-- WORK HERE
│   ├── base.py
│   ├── affordance_type1.py
│   ├── affordance_type2.py
│   └── visual_trace.py
└── job_server/          # Distributed job system
    ├── server.py        # FastAPI REST server
    ├── worker.py        # Base worker class
    ├── pipeline_worker.py
    └── client.py        # CLI client
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

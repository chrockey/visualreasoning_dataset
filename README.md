# Visual Reasoning Annotation

Pipeline system for visual reasoning dataset annotation using Molmo and SAM2.

## Installation

```bash
./install.sh
```

## Supported Pipelines

| Pipeline                    | Use Case                                   | Command                                             |
|-----------------------------|--------------------------------------------|-----------------------------------------------------|
| **GT Visual Trace Droid**    | With GT pose (3D→2D projection)             | `python -m src.pipelines.gt_visual_trace_droid`     |
| **Visual Trace Language Table** | Without GT pose (SAM3 video tracking) | `python -m src.pipelines.affordance_type_language_table` |
| **Visual Trace Bridge**      | Without GT pose (SAM3 video tracking with Gemini) | `python -m src.pipelines.affordance_type_bridge`    |



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
# data['descriptions]: 
# [(0, 287, 'Add lids onto four cups placed on a wooden table with a red background.')]
# data['metadata']: camera, MANO hand poses, transforms

# Open X-Embodiment: Robot manipulation episodes
dataset = OXEDataset()  # Default: vla-dataset-samples/open-x-embodiment
print(f"Episodes: {len(dataset)}")
data = dataset[0]
# data['video_name']: "asu_table_top_converted_externally_to_rlds/00003/0"
#                     {dataset_name}/{shard_id}/{episode_in_shard}
# data['frames']: (3225, 256, 256, 3)
# data['descriptions]: 
# [(0, 3224, 'Interact with the objects in diverse but meaningful ways.')]
# data['metadata']['state']: robot joint states
# data['metadata']['action']: robot actions
# data['metadata']['language_embedding']: 512-dim embedding
# data['metadata']['tfrecord_info']: shard tracking info for annotations

# AgiBotWorld-Beta: Bimanual manipulation dataset
dataset = AgiBotWorldDataset()
print(f"Episodes: {len(dataset)}")
data = dataset[0]
# data['frames']: (1295, 480, 640, 3)
# data['descriptions']:
# [(36, 187, 'Retrieve cucumber from the shelf.'),
#  (187, 426, 'Place the held cucumber into the plastic bag in the shopping cart.'),
#  (426, 591, 'Retrieve tomato from the shelf.'),
#  (591, 788, 'Place the held tomato into the plastic bag in the shopping cart.'),
#  (788, 956, 'Retrieve corn from the shelf.'),
#  (956, 1232, "Place the held corn into the shopping cart's plastic bag.")]
# data['metadata']['hand_left_frames]: frames from hand-left cam
# data['metadata']['hand_right_frames]: frames from hand-right cam
# data['metadata']['action_config']: action text, skill(pick,place,..)
# data['metadata']['proprio_stats]: effector (orientation, velocity, ..)

# HoloAssist: Egocentric human interaction dataset
dataset = HoloAssistDataset()
print(f"Videos: {len(dataset)}")
data = dataset[0]
# data['frames]: (9933, 504, 896, 3)
# data['descriptions']
# [(268, 691, 'The student grabs the GoPro.'),
#  (731, 1620, 'The student changes the battery for the GoPro.'),
#  (1672, 7444, 'The student opens the GoPro.'),
#  (7496, 7667, 'The student turns on their GoPro.'),
#  (7685, 7957, 'The student turns off the gopro.'),
#  (7988, 8604, 'The student assembles the mounting_peg.'),
#  (8679, 8864, 'The student disassemble the mounting_peg.'),
#  (8883, 9526, 'The student assemble handheld_grip.'),
#  (9543, 9896, 'The students disassemble the handheld_grip.')]
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
<details open>
<summary>Visualize GT robot gripper trajectories</summary>

- MP4 files are saved under viz/*
- Runnable datasets
    - [x] AgiBotWorld
    - [ ] EgoDex
    - [ ] HoloAssist
    - [ ] Open X-Embodiment
</details>

```bash
python -m src.pipelines.gt_visual_trace
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

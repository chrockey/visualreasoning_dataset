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
| **Visual Trace Language Table** | Without GT pose (SAM3 video tracking) | `python -m   src.pipelines.affordance_type_language_table` |
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

## Dataset Structure

### 1️⃣ DROID (GT-based Visual Trace)
```
# Download the DROID raw dataset (required)
gsutil -m cp -r gs://gresearch/robotics/droid_raw <path_to_your_target_dir>

droid_dataset/
└── date/                             # Recording date
    └── recordings/
        ├── MP4/
        │   ├── 18026681.mp4          # Gripper-mounted camera
        │   ├── 22008760.mp4          # External camera (viewpoint 1)
        │   └── 24400334.mp4          # External camera (viewpoint 2)
        ├── SVO/                      # ZED camera recordings (for intrinsics extraction)
        │   ├── 18026681.svo
        │   ├── 22008760.svo
        │   └── 24400334.svo
        └── trajectory.h5             # Camera extrinsic parameters (trajectory)
```
### 2️⃣ Language Table (SAM3, No GT Pose)
```
language_table_dataset/
├── language_table-train.tfrecord-00000-of-01024      # Sharded TFRecord file containing training episodes
├── dataset_info.json
├── dataset_statistics_*.json
└── features.json                                     # Features(Observation , Action(2D Cartesian))
```

### 3️⃣ Bridge (SAM3 + Gemini, No GT Pose)
```
bridge_folder/
├── bridge_oxe.tfrecord-00000-of-01024                # Sharded TFRecord file containing training episodes
├── dataset_info.json
├── dataset_statistics_*.json
└── features.json                                     # Features(Observation, Action(3D translation + 3D rotation))
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
│   ├── affordance_type_bridge.py
│   ├── affordance_type_language_table.py
│   └── gt_visual_trace_droid.py
└── job_server/          # Distributed job system
    ├── server.py        # FastAPI REST server
    ├── worker.py        # Base worker class
    ├── pipeline_worker.py
    └── client.py        # CLI client
```

## Testing Pipelines

```bash
python -m src.pipelines.affordance_type_language_table
python -m src.pipelines.affordance_type_bridge
```
<details open>
<summary>Visualize GT robot gripper trajectories</summary>

- MP4 files are saved under viz_video/*
- Runnable datasets
    - [ ] AgiBotWorld
    - [ ] EgoDex
    - [ ] HoloAssist
    - [x] Open X-Embodiment
</details>

```bash
python -m src.pipelines.gt_visual_trace_droid
``` 


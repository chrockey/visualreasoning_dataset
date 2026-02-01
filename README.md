# Visual Reasoning Annotation

Pipeline system for visual reasoning dataset annotation using SAM3 Tracking and Gemini.

> **Note:** The SAM3 Git module has been modified for this project; please refer to the code included in this repository rather than the upstream implementation.

## Installation

```bash
./install.sh
```

## Pipelines

### 1. GT Visual Trace
Uses ground-truth camera parameters to project 3D trajectories to 2D pixel coordinates.

| Supported Dataset | Config | Command |
|-------------------|--------|---------|
| DROID | `gt_visual_trace_droid.yaml` | `python -m src.pipelines.gt_visual_trace.droid` |
| AgiBotWorld | `gt_visual_trace_agibotworld.yaml` | `python -m src.pipelines.gt_visual_trace.agibotworld` |

### 2. SAM3 Tracker
Uses SAM3 video tracking without ground-truth pose. Dataset-agnostic.

| Supported Dataset | Config | Command |
|-------------------|--------|---------|
| Language Table | `visual_trace_language_table.yaml` | `python -m src.pipelines.visual_trace.sam3_tracker --config visual_trace_language_table --data-dir ./data --episode-index 0` |

### 3. Gemini + SAM3 Tracker
Uses Gemini for gripper detection refinement + SAM3 video tracking. Dataset-agnostic.

| Supported Dataset | Config | Command |
|-------------------|--------|---------|
| Bridge | `visual_trace_bridge.yaml` | `python -m src.pipelines.visual_trace.gemini_sam3_tracker --config visual_trace_bridge --data-dir ./data --episode-index 0` |

## Dataset Structure

#### DROID
```bash
# Download
gsutil -m cp -r gs://gresearch/robotics/droid_raw <path>
```
```
droid_dataset/
└── date/
    └── recordings/
        ├── MP4/
        │   ├── 18026681.mp4      # Gripper-mounted camera
        │   ├── 22008760.mp4      # External camera 1
        │   └── 24400334.mp4      # External camera 2
        ├── SVO/                  # ZED recordings (for intrinsics)
        │   └── *.svo
        └── trajectory.h5         # Camera extrinsics
```

#### AgiBotWorld
```
agibotworld_dataset/
└── episode/
    ├── head_camera.mp4
    ├── hand_left_camera.mp4
    ├── hand_right_camera.mp4
    └── metadata.json             # Camera parameters
```

#### Language Table / Bridge (OXE TFRecord format)
```
dataset_folder/
├── *.tfrecord-*-of-*
├── dataset_info.json
└── features.json
```

## Project Structure

```
src/
├── datasets/
│   ├── base.py
│   ├── agibotworld.py
│   └── oxe.py
├── models/
│   └── sam3_video_tracker.py
├── pipelines/
│   ├── base.py
│   ├── gt_visual_trace/          # GT-based (dataset-specific)
│   │   ├── droid.py
│   │   └── agibotworld.py
│   └── visual_trace/             # SAM3-based (dataset-agnostic)
│       ├── sam3_tracker.py
│       └── gemini_sam3_tracker.py
├── utils/
│   └── gt_visual_trace.py
└── job_server/
    ├── server.py
    ├── pipeline_worker.py
    └── client.py
```

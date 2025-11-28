from typing import Optional

import imageio
import numpy as np
import torch

def parse_query_points(query_points: str) -> torch.Tensor:
    """Parse semicolon-separated t,x,y triplets into a tensor."""
    points = []
    for raw_point in query_points.split(":"):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        values = [value.strip() for value in raw_point.split(",")]
        if len(values) != 3:
            raise ValueError(
                f"Each query point must have three values (t,x,y), got: {raw_point}"
            )
        t, x, y = values
        points.append([int(t), float(x), float(y)])
    if not points:
        raise ValueError("No valid query points were provided.")
    return torch.tensor(points, dtype=torch.float32)[None]

def read_video_from_path(path, num_frames=None) -> torch.Tensor:
    """Load video frames and return a tensor shaped (T, H, W, C)."""
    try:
        reader = imageio.get_reader(path)
    except Exception as exc:
        raise RuntimeError(f"Error opening video file {path}: {exc}") from exc

    frames = []
    for idx, frame in enumerate(reader):
        if num_frames is not None and idx >= num_frames:
            break
        frame_np = np.asarray(frame)
        if frame_np.ndim != 3 or frame_np.shape[2] != 3:
            raise ValueError(
                f"Each frame must have shape (H, W, 3); got frame with shape {frame_np.shape}"
            )
        frames.append(frame_np)

    reader.close()

    if not frames:
        raise RuntimeError(f"No frames were read from video {path}")

    video = np.stack(frames, axis=0)
    return torch.from_numpy(video)

class CoTracker:
    """Thin wrapper that loads a CoTracker model and exposes a callable interface."""

    def __init__(
        self,
        model_id: str = "facebookresearch/co-tracker:cotracker3_offline",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        repo, name = model_id.split(":", 1)
        self.model = torch.hub.load(repo, name).to(self.device)

    def __call__(
        self,
        video: torch.Tensor,
        queries: Optional[torch.Tensor],
        grid_size: int = 10,
        grid_query_frame: int = 0,
    ):
        video = self.check_dim(video).to(self.device)
        pred_tracks, pred_visibility = self.model(
            video,
            queries=queries,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
        
        pred_tracks, pred_visibility = pred_tracks.cpu(), pred_visibility.cpu()
        return (
            pred_tracks,       # (B, T, N, 2) track coordinates over time
            pred_visibility,   # (B, T, N) visibility/confidence per track
        )

    def check_dim(self, video: torch.Tensor) -> torch.Tensor:
        """Ensure the video tensor is shaped (1, T, C, H, W)."""
        if video.ndim != 4:
            raise ValueError(f"Video tensor must have shape (T, H, W, C), got {video.shape}")
        assert (
            video.shape[-1] == 3
        ), f"Video channel dimension must be 3, got {video.shape[-1]}"
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
        video = video.unsqueeze(0)  # (1, T, C, H, W)
        return video.float()

if __name__ == "__main__":
    import argparse
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--query_points",
        type=str,
        default=None,
        help="Semicolon separated list of query points formatted as t,x,y:t,x,y",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    args = parser.parse_args()
    
    video = read_video_from_path(args.video_path)
    
    if args.query_points:
        query_points = parse_query_points(args.query_points).to(device=device, dtype=torch.float32)
    else:
        query_points = None
    
    model = CoTracker()
    pred_tracks, pred_visibility = model(
        video,
        queries=query_points,
        grid_size=0 if args.query_points is not None else args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )
    
    print(pred_tracks)
    print(pred_visibility)

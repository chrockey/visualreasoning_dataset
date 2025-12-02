from typing import Optional
import numpy as np
import torch
from scipy import ndimage

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
        queries = self.correct_keypoints(queries, timestep=grid_query_frame)
        
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

    def correct_keypoints(self, keypoints: torch.Tensor, timestep=0) -> torch.Tensor:
        """Correct keypoints to be within the image boundaries."""
        # keypoint (t, x, y)
        # correct x and y to be within the image boundaries
        # (x, y) -> (t, x, y)
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints).to(self.device)
        if keypoints.shape[2] == 2:
            keypoints = torch.cat([torch.ones(keypoints.shape[0], keypoints.shape[1], 1).to(self.device) * timestep, keypoints], dim=2)
        return keypoints

    def check_dim(self, video: torch.Tensor) -> torch.Tensor:
        """Ensure the video tensor is shaped (1, T, C, H, W)."""
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video).to(self.device)
        if video.ndim != 4:
            raise ValueError(f"Video tensor must have shape (T, H, W, C), got {video.shape}")
        assert (
            video.shape[-1] == 3
        ), f"Video channel dimension must be 3, got {video.shape[-1]}"
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
        video = video.unsqueeze(0)  # (1, T, C, H, W)
        return video.float()
    
    @staticmethod
    def _get_interior_mask(mask: np.ndarray) -> np.ndarray:
        """Get interior region of mask by applying erosion.
        Args:
            mask: Binary mask as numpy array (H, W)
        Returns:
            Interior mask with boundary points removed
        """
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        kernel = np.ones((5, 5), dtype=bool)
        return ndimage.binary_erosion(mask, structure=kernel)
    
    @staticmethod
    def extract_keypoints_from_mask(mask: np.ndarray) -> np.ndarray:
        """Extract 3 keypoints from a single mask.
        Args:
            mask: Binary mask as numpy array (H, W) with values 0 or 1
        Returns:
            Keypoints as numpy array (3, 2) with (x, y) coordinates
        """
        # Get interior mask (away from boundary)
        interior_mask = CoTracker._get_interior_mask(mask)
        
        # Get all points inside the interior mask
        y_coords, x_coords = np.where(interior_mask)
        if len(y_coords) == 0:
            # If erosion removed everything, fall back to original mask
            # but use centroid only (safer)
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                # Empty mask, return zeros
                return np.zeros((3, 2), dtype=np.float32)
            points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
            centroid = points.mean(axis=0)
            # Return centroid repeated 3 times if we can't find interior points
            return np.tile(centroid, (3, 1))
        
        points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
        
        # Calculate centroid from interior points
        centroid = points.mean(axis=0)
        
        # First keypoint: farthest from centroid
        distances_from_centroid = np.linalg.norm(points - centroid, axis=1)
        first_idx = np.argmax(distances_from_centroid)
        first_point = points[first_idx]
        
        # Second keypoint: farthest from both centroid and first point
        # Use minimum distance to already selected points
        dist_to_centroid = np.linalg.norm(points - centroid, axis=1)
        dist_to_first = np.linalg.norm(points - first_point, axis=1)
        min_distances = np.minimum(dist_to_centroid, dist_to_first)
        second_idx = np.argmax(min_distances)
        second_point = points[second_idx]
        
        # Combine: centroid + 2 farthest points
        keypoints = np.vstack([centroid, first_point, second_point])
        
        return keypoints
    
    @staticmethod
    def extract_keypoints_from_masks(masks: np.ndarray) -> np.ndarray:
        """Extract keypoints from all masks.
        Args:
            masks: Binary masks as numpy array (n, H, W)
        Returns:
            Keypoints as numpy array (n, 3, 2) where each mask has 3 keypoints (x, y)
        """
        keypoints_list = []
        for i in range(masks.shape[0]):
            keypoints = CoTracker.extract_keypoints_from_mask(masks[i])
            keypoints_list.append(keypoints)
        
        return np.stack(keypoints_list, axis=0)


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
    import imageio

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


class KeypointFilter:
    # Rule-based filtering system for tracked keypoints
    def __init__(self, traj_top_k : int = 3):
        self.traj_top_k = traj_top_k

    @staticmethod
    def __top_moving_keypoints__(tracked_keypoints: np.ndarray, tracked_visibility: np.ndarray, top_k: int = 3):
        B, T, N, _ = tracked_keypoints.shape

        if N <= top_k:
            print(f"Number of keypoints ({N}) is less than or equal to top_k ({top_k}). No filtering applied.")
            return tracked_keypoints, tracked_visibility, np.arange(N)

        # Calculate total displacement for each keypoint from start to end frame
        displacements = np.zeros((B, N))

        for b in range(B):
            for i in range(N):
                start_pos = tracked_keypoints[b, 0, i]  # First frame
                end_pos = tracked_keypoints[b, -1, i]   # Last frame
                displacement = np.linalg.norm(end_pos - start_pos)
                displacements[b, i] = displacement

        # Average displacement across batch
        avg_displacements = displacements.mean(axis=0)

        # Get indices of top-k keypoints with largest displacement
        # Use copy() to avoid negative stride issues
        top_indices = np.argsort(avg_displacements)[::-1][:top_k].copy()

        # Filter keypoints and visibility
        filtered_keypoints = tracked_keypoints[:, :, top_indices, :]
        filtered_visibility = tracked_visibility[:, :, top_indices]

        return filtered_keypoints, filtered_visibility, len(top_indices)

    def __call__(self, tracked_keypoints: np.ndarray, tracked_visibility: np.ndarray):
        tracked_keypoints, tracked_visibility, keypoint_num = self.__top_moving_keypoints__(
            tracked_keypoints, tracked_visibility, top_k=self.traj_top_k
        )
        return tracked_keypoints, tracked_visibility, keypoint_num
    

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
from typing import Dict, Any
import numpy as np
import cv2


def draw_sliding_trajectory(frame, uv_all, current_idx, window,
                            line_color, point_color, point_radius=5, line_thickness=2):
    """
    Draw trajectory with gradient effect from older (lighter) to current (darker/brighter).
    The gradient uses the same color family as point_color.
    """
    start = max(0, current_idx - window + 1)
    num_points = current_idx - start + 1

    prev = None
    for idx, k in enumerate(range(start, current_idx + 1)):
        uv = uv_all[k]
        if uv is None:
            prev = None
            continue

        if prev is not None:
            # Calculate gradient: older segments are lighter (fade towards white)
            # alpha ranges from 0.3 (oldest) to 1.0 (newest)
            alpha = 0.3 + 0.7 * (idx / max(1, num_points - 1))

            # Blend point_color with white for gradient effect
            # point_color is in BGR format (B, G, R)
            gradient_color = tuple(
                int(point_color[i] * alpha + 255 * (1 - alpha)) for i in range(3)
            )

            cv2.line(frame, prev, uv, gradient_color, line_thickness)
        prev = uv

    # Draw current point with black border
    if uv_all[current_idx] is not None:
        # Black border (outer circle)
        cv2.circle(frame, uv_all[current_idx], point_radius + 2, (0, 0, 0), -1)
        # Colored point (inner circle)
        cv2.circle(frame, uv_all[current_idx], point_radius, point_color, -1)


def load_intrinsic_from_dict(intr_dict):
    """
    Load intrinsic parameters from a dictionary (from metadata).
    """
    fx = float(intr_dict["fx"])
    fy = float(intr_dict["fy"])
    cx = float(intr_dict["ppx"])
    cy = float(intr_dict["ppy"])
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K, intr_dict


def load_extrinsic_sequence_from_list(extr_list):
    """
    Load extrinsic sequence from a list of extrinsic dicts (from metadata).
    Returns R_list, t_list where each R is (3,3) rotation matrix
    and each t is (3,1) translation vector.
    """
    R_list, t_list = [], []
    for ext_data in extr_list:
        # ext_data is already the extrinsic dict (unwrapped by dataset loader)
        R = np.array(ext_data["rotation_matrix"], dtype=np.float64)
        t = np.array(ext_data["translation_vector"], dtype=np.float64).reshape(3, 1)
        R_list.append(R)
        t_list.append(t)
    return R_list, t_list


def project_camera_trajectories_to_2d(
    metadata: Dict[str, Any],
    num_frames: int,
    viz_width: int = 640,
    viz_height: int = 480,
):
    """
    Project 3D camera trajectories to 2D pixel coordinates.

    Args:
        metadata: Metadata dict containing camera_params with:
            - head_extrinsics: List of head camera extrinsics
            - hand_left_extrinsics: List of left hand extrinsics
            - hand_right_extrinsics: List of right hand extrinsics
            - head_intrinsics: Head camera intrinsic parameters
        num_frames: Number of frames to process
        viz_width: Target visualization width in pixels
        viz_height: Target visualization height in pixels

    Returns:
        Dictionary containing projected 2D trajectories:
        {
            "head_2d": (T, 2) array,       # Head camera trajectory in pixel coordinates
            "left_hand_2d": (T, 2) array,  # Left hand trajectory in pixel coordinates
            "right_hand_2d": (T, 2) array, # Right hand trajectory in pixel coordinates
            "metadata": {
                "num_frames": int,          # Number of frames
                "video_width": int,         # Video width in pixels
                "video_height": int,        # Video height in pixels
                "format": str,              # "uv_coordinates"
                "nan_meaning": str,         # "invisible" (out of bounds or behind camera)
                "coordinate_system": str    # "image" (top-left origin, x-right, y-down)
            }
        }
        Returns None if camera parameters are missing.
    """
    camera_params = metadata.get("camera_params", {})

    # Extract camera parameters from metadata
    head_extrinsics = camera_params.get("head_extrinsics", [])
    left_extrinsics = camera_params.get("hand_left_extrinsics", [])
    right_extrinsics = camera_params.get("hand_right_extrinsics", [])
    head_intrinsics = camera_params.get("head_intrinsics")

    if not head_extrinsics or not left_extrinsics or not right_extrinsics:
        print(f"[WARNING] Missing camera extrinsics, skipping camera trajectory video")
        return None

    if head_intrinsics is None:
        print(f"[WARNING] Missing head camera intrinsics, skipping camera trajectory video")
        return None

    # Load camera parameters
    R_head, t_head = load_extrinsic_sequence_from_list(head_extrinsics)
    R_left, t_left = load_extrinsic_sequence_from_list(left_extrinsics)
    R_right, t_right = load_extrinsic_sequence_from_list(right_extrinsics)
    K_head, _ = load_intrinsic_from_dict(head_intrinsics)

    fx_h = K_head[0, 0]
    fy_h = K_head[1, 1]
    cx_h = K_head[0, 2]
    cy_h = K_head[1, 2]

    # Calculate trajectories in world coordinates
    left_traj = np.stack([t_left[i].reshape(3) for i in range(len(R_left))], axis=0)
    right_traj = np.stack([t_right[i].reshape(3) for i in range(len(R_right))], axis=0)
    head_centers = np.stack([t_head[i].reshape(3) for i in range(len(R_head))], axis=0)

    # Determine total length
    T_data = max(len(R_left), len(R_right))
    n_head = len(R_head)
    T = min(n_head, T_data, num_frames)

    # Slice to total length
    left_traj = left_traj[:T]
    right_traj = right_traj[:T]
    head_centers = head_centers[:T]
    R_head = R_head[:T]
    t_head = t_head[:T]

    traj_head_3d = head_centers
    traj_left_3d = left_traj
    traj_right_3d = right_traj

    # Project trajectories to 2D head camera plane
    def proj_to_head_cam(P_world, R_h, t_h):
        if P_world is None:
            return None
        Pw = P_world.reshape(3)
        th = t_h.reshape(3)
        Pc = R_h.T @ (Pw - th)
        X, Y, Z = Pc
        if Z <= 1e-6:
            return None
        u = fx_h * (X / Z) + cx_h
        v = fy_h * (Y / Z) + cy_h
        if 0 <= u < viz_width and 0 <= v < viz_height:
            return (int(u), int(v))
        else:
            return None

    head_uv_2d = []
    left_uv_2d = []
    right_uv_2d = []

    for i in range(T):
        R_h = R_head[i]
        t_h = t_head[i]
        head_uv_2d.append(proj_to_head_cam(traj_head_3d[i], R_h, t_h))
        left_uv_2d.append(proj_to_head_cam(traj_left_3d[i], R_h, t_h))
        right_uv_2d.append(proj_to_head_cam(traj_right_3d[i], R_h, t_h))

    # Convert list of tuples/None to numpy arrays with NaN for None values
    # Format: (T, 2) array where T is number of frames
    # Values are (u, v) pixel coordinates in the output video frame
    # None values (invisible/out-of-bounds points) are converted to NaN
    def convert_to_array(uv_list):
        """Convert list of (u,v) tuples or None to (T, 2) numpy array with NaN for None."""
        arr = np.full((len(uv_list), 2), np.nan, dtype=np.float32)
        for i, uv in enumerate(uv_list):
            if uv is not None:
                arr[i] = uv
        return arr

    # Prepare return data with projected 2D trajectories
    trajectory_data = {
        # Projected 2D trajectories (T, 2) in pixel coordinates
        "head_2d": convert_to_array(head_uv_2d),      # Head camera trajectory (red point in video)
        "left_hand_2d": convert_to_array(left_uv_2d),  # Left hand trajectory (green point in video)
        "right_hand_2d": convert_to_array(right_uv_2d), # Right hand trajectory (blue point in video)

        # Metadata for interpreting the trajectories
        "metadata": {
            "num_frames": T,                # Number of frames in trajectory
            "video_width": viz_width,       # Output video width in pixels
            "video_height": viz_height,     # Output video height in pixels
            "format": "uv_coordinates",     # Coordinate format: (u, v) pixel coordinates
            "nan_meaning": "invisible",     # NaN indicates point is invisible (out of bounds or behind camera)
            "coordinate_system": "image",   # Coordinate system: top-left origin, x-right, y-down
        }
    }

    return trajectory_data


def generate_trajectory_visualization_video(
    video_frames: np.ndarray,
    trajectory_data: Dict[str, Any],
    output_path: str,
    trace_window: int = 60,
):
    """
    Generate visualization video from 2D trajectory data.

    Args:
        video_frames: Full video frames (head camera), RGB format (T, H, W, 3)
        trajectory_data: Dictionary containing projected 2D trajectories:
            - "head_2d": (T, 2) array with head camera trajectory
            - "left_hand_2d": (T, 2) array with left hand trajectory
            - "right_hand_2d": (T, 2) array with right hand trajectory
            - "metadata": Dict with video_width, video_height, num_frames
        output_path: Output video file path
        trace_window: Number of frames to show in trajectory trace
    """
    # Extract trajectory arrays
    head_2d = trajectory_data["head_2d"]      # (T, 2) array
    left_2d = trajectory_data["left_hand_2d"]  # (T, 2) array
    right_2d = trajectory_data["right_hand_2d"] # (T, 2) array

    # Extract metadata
    meta = trajectory_data["metadata"]
    viz_width = meta["video_width"]
    viz_height = meta["video_height"]
    T = meta["num_frames"]

    # Convert numpy arrays back to list of tuples/None for drawing
    def array_to_list(arr):
        """Convert (T, 2) array to list of (u, v) tuples or None."""
        uv_list = []
        for i in range(len(arr)):
            if np.isnan(arr[i, 0]) or np.isnan(arr[i, 1]):
                uv_list.append(None)
            else:
                uv_list.append((int(arr[i, 0]), int(arr[i, 1])))
        return uv_list

    head_uv_2d = array_to_list(head_2d)
    left_uv_2d = array_to_list(left_2d)
    right_uv_2d = array_to_list(right_2d)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 30, (viz_width, viz_height))

    print(f"[INFO] Generating camera trajectory video: {output_path}")

    # Generate frames
    for i in range(T):
        # Get video frame and convert RGB to BGR
        if i < len(video_frames):
            frame = video_frames[i].copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (viz_width, viz_height))
        else:
            frame = np.ones((viz_height, viz_width, 3), np.uint8) * 255

        # Draw trajectories
        # Head trajectory (red point)
        draw_sliding_trajectory(frame, head_uv_2d, i, trace_window,
                                line_color=(0, 0, 0),
                                point_color=(0, 0, 255),
                                point_radius=8,
                                line_thickness=4)
        # Left hand trajectory (green point)
        draw_sliding_trajectory(frame, left_uv_2d, i, trace_window,
                                line_color=(255, 0, 255),
                                point_color=(0, 255, 0),
                                point_radius=8,
                                line_thickness=4)
        # Right hand trajectory (blue point)
        draw_sliding_trajectory(frame, right_uv_2d, i, trace_window,
                                line_color=(0, 0, 255),
                                point_color=(255, 0, 0),
                                point_radius=8,
                                line_thickness=4)

        # Frame info
        cv2.putText(frame, f"Frame {i+1}/{T}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        writer.write(frame)

    writer.release()
    print(f"[INFO] Saved camera trajectory video: {output_path}")


def generate_camera_trajectory_video(
    video_frames: np.ndarray,
    metadata: Dict[str, Any],
    output_path: str,
    trace_window: int = 60,
    viz_width: int = 640,
    viz_height: int = 480,
):
    """
    Generate 2D camera trajectory visualization video (combines projection + video generation).

    This is a convenience function that combines:
    1. project_camera_trajectories_to_2d() - Projects 3D trajectories to 2D
    2. generate_trajectory_visualization_video() - Generates video from 2D data

    Args:
        video_frames: Full video frames (head camera), RGB format (T, H, W, 3)
        metadata: Metadata dict containing camera_params
        output_path: Output video path
        trace_window: Number of frames to show in trajectory trace
        viz_width: Output video width
        viz_height: Output video height

    Returns:
        Dictionary containing projected 2D trajectories (see project_camera_trajectories_to_2d)
        Returns None if camera parameters are missing.
    """
    num_frames = len(video_frames)

    # Step 1: Project 3D trajectories to 2D
    trajectory_data = project_camera_trajectories_to_2d(
        metadata=metadata,
        num_frames=num_frames,
        viz_width=viz_width,
        viz_height=viz_height,
    )

    if trajectory_data is None:
        return None

    # Step 2: Generate visualization video from 2D trajectory data
    generate_trajectory_visualization_video(
        video_frames=video_frames,
        trajectory_data=trajectory_data,
        output_path=output_path,
        trace_window=trace_window,
    )

    return trajectory_data
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import numpy as np
import cv2
import json
# DROID / ZED
import h5py
import pyzed.sl as sl


# ============================================================
# Drawing utilities
# ============================================================
def draw_sliding_trajectory(
    frame,
    uv_all,
    current_idx,
    window,
    line_color,
    point_color,
    point_radius=5,
    line_thickness=2,
):
    """
    Draw a sliding-window trajectory with a temporal fade effect.
    Older segments are lighter, recent segments are darker.
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
            # English note: Alpha increases with time, making newer segments darker.
            alpha = 0.3 + 0.7 * (idx / max(1, num_points - 1))
            gradient_color = tuple(
                int(point_color[i] * alpha + 255 * (1 - alpha)) for i in range(3)
            )
            cv2.line(frame, prev, uv, gradient_color, line_thickness)

        prev = uv

    if uv_all[current_idx] is not None:
        cv2.circle(frame, uv_all[current_idx], point_radius + 2, (0, 0, 0), -1)
        cv2.circle(frame, uv_all[current_idx], point_radius, point_color, -1)

# ============================================================
# Save Json log per frame
# ============================================================

def save_frame_log(
    log_dir: Path,
    frame_idx: int,
    uv: Optional[Tuple[int, int]],
    fps: float,
):
    log_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "frame_idx": frame_idx,
        "timestamp_sec": frame_idx / fps,
        "hand_uv": list(uv) if uv is not None else None,
        "visible": uv is not None,
    }

    log_path = log_dir / f"frame_{frame_idx:06d}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


# ============================================================
# Common camera utilities
# ============================================================
def load_intrinsic_from_dict(intr_dict):
    """
    Load camera intrinsics from metadata dictionary.
    """
    fx = float(intr_dict["fx"])
    fy = float(intr_dict["fy"])
    cx = float(intr_dict["ppx"])
    cy = float(intr_dict["ppy"])
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def load_extrinsic_sequence_from_list(extr_list):
    """
    Load a sequence of extrinsics into rotation / translation lists.
    """
    R_list, t_list = [], []
    for ext in extr_list:
        R_list.append(np.array(ext["rotation_matrix"], dtype=np.float64))
        t_list.append(np.array(ext["translation_vector"], dtype=np.float64).reshape(3, 1))
    return R_list, t_list


# ============================================================
# DROID-specific helpers
# ============================================================
def _load_pose_seq(h5_path: Path, key: str) -> np.ndarray:
    """
    Load pose sequence from trajectory.h5.
    Pose format: [x, y, z, rx, ry, rz]
    """
    with h5py.File(str(h5_path), "r") as f:
        if key not in f:
            raise KeyError(f"[H5] key not found: {key}")
        arr = np.array(f[key], dtype=np.float64)

    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"[H5] Expected (T,6), got {arr.shape}")
    return arr


def _euler_xyz_to_R(rx, ry, rz) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def _pose6_to_T(pose6, rot_mode: str) -> np.ndarray:
    """
    Convert a 6-DoF pose to SE(3) matrix.
    """
    if rot_mode != "euler_xyz":
        raise ValueError("Only euler_xyz rotation is supported")

    x, y, z, rx, ry, rz = pose6
    T = np.eye(4)
    T[:3, :3] = _euler_xyz_to_R(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


def _world_to_cam(Pw, T_world_cam):
    """
    Transform a world point into camera coordinates.
    """
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    return R.T @ (Pw - t)


def _project_point(Pc, fx, fy, cx, cy, W, H) -> Optional[Tuple[int, int]]:
    """
    Project a 3D camera-frame point into image coordinates.
    """
    X, Y, Z = Pc
    if Z <= 1e-6:
        return None

    u = cx + fx * (X / Z)
    v = cy + fy * (Y / Z)

    if 0 <= u < W and 0 <= v < H:
        return int(u), int(v)
    return None


def _get_intrinsic_from_svo(svo_path: Path, eye: str) -> Tuple[float, float, float, float]:
    """
    Extract fx, fy, cx, cy from a ZED SVO file.
    """
    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO: {svo_path}")

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    cam = calib.left_cam if eye == "left" else calib.right_cam

    fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
    zed.close()
    return fx, fy, cx, cy


# ============================================================
# Projection entry point (hand-only version)
# ============================================================
def project_camera_trajectories_to_2d(
    metadata: Dict[str, Any],
    num_frames: int,
    viz_width: int = 640,
    viz_height: int = 480,
):
    """
    Project camera trajectories into 2D image space.

    Output format:
      - head_2d: (T,2)  (may be NaN if unused)
      - hand_2d: (T,2)  single gripper / hand trajectory
    """

    # --------------------------------------------------------
    # DROID GT trace mode
    # --------------------------------------------------------
    if "droid_gt_trace" in metadata:
        cfg = metadata["droid_gt_trace"]

        traj_h5 = Path(cfg["traj_h5"])
        svo_path = Path(cfg["svo_path"])
        view_cam_id = cfg["view_cam_id"]
        arm_cam_id = cfg["arm_cam_id"]
        gripper_offset_id = cfg["gripper_offset_id"]
        eye = cfg.get("eye", "left")
        rot_mode = cfg.get("rot_mode", "euler_xyz")

        fx, fy, cx, cy = _get_intrinsic_from_svo(svo_path, eye)

        view_pose = _load_pose_seq(traj_h5, f"observation/camera_extrinsics/{view_cam_id}")
        arm_pose = _load_pose_seq(traj_h5, f"observation/camera_extrinsics/{arm_cam_id}")
        off_pose = _load_pose_seq(traj_h5, f"observation/camera_extrinsics/{gripper_offset_id}")

        T = min(len(view_pose), len(arm_pose), len(off_pose), num_frames)

        hand_uv: List[Optional[Tuple[int, int]]] = []

        for i in range(T):
            T_w_arm = _pose6_to_T(arm_pose[i], rot_mode)
            T_arm_grip = _pose6_to_T(off_pose[i], rot_mode)
            Pw_grip = (T_w_arm @ T_arm_grip)[:3, 3]

            T_w_view = _pose6_to_T(view_pose[i], rot_mode)
            Pc = _world_to_cam(Pw_grip, T_w_view)

            hand_uv.append(_project_point(Pc, fx, fy, cx, cy, viz_width, viz_height))

        hand_2d = np.full((T, 2), np.nan, dtype=np.float32)
        for i, uv in enumerate(hand_uv):
            if uv is not None:
                hand_2d[i] = uv

        return {
            "head_2d": np.full_like(hand_2d, np.nan),
            "hand_2d": hand_2d,
            "metadata": {
                "num_frames": T,
                "video_width": viz_width,
                "video_height": viz_height,
                "format": "uv_coordinates",
                "nan_meaning": "invisible",
                "coordinate_system": "image",
            },
        }

    # --------------------------------------------------------
    # camera_params mode (single hand)
    # --------------------------------------------------------
    camera_params = metadata.get("camera_params", {})
    if not camera_params:
        return None

    head_extr = camera_params.get("head_extrinsics", [])
    hand_extr = camera_params.get("hand_extrinsics", [])
    intr = camera_params.get("head_intrinsics")

    if not head_extr or not hand_extr or intr is None:
        return None

    R_h, t_h = load_extrinsic_sequence_from_list(head_extr)
    R_hand, t_hand = load_extrinsic_sequence_from_list(hand_extr)
    K = load_intrinsic_from_dict(intr)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    T = min(len(R_h), len(R_hand), num_frames)
    hand_uv = []

    for i in range(T):
        Pw = t_hand[i].reshape(3)
        Pc = R_h[i].T @ (Pw - t_h[i].reshape(3))
        hand_uv.append(_project_point(Pc, fx, fy, cx, cy, viz_width, viz_height))

    hand_2d = np.full((T, 2), np.nan, dtype=np.float32)
    for i, uv in enumerate(hand_uv):
        if uv is not None:
            hand_2d[i] = uv

    return {
        "head_2d": np.full_like(hand_2d, np.nan),
        "hand_2d": hand_2d,
        "metadata": {
            "num_frames": T,
            "video_width": viz_width,
            "video_height": viz_height,
            "format": "uv_coordinates",
            "nan_meaning": "invisible",
            "coordinate_system": "image",
        },
    }


# ============================================================
# Video generation (hand-only visualization)
# ============================================================
def generate_trajectory_visualization_video(
    video_frames: np.ndarray,
    trajectory_data: Dict[str, Any],
    output_path: str,
    trace_window: int = 60,
):
    """
    Generate visualization video from 2D trajectory data.

    Args:
        video_frames: Full video frames (RGB) in shape (T, H, W, 3).
        trajectory_data: Dictionary containing projected 2D trajectories:
            - "hand_2d": (T, 2) array
            - "metadata": Dict with video_width, video_height, num_frames
        output_path: Output video file path
        trace_window: Number of frames to show in trajectory trace
    """
    hand_2d = trajectory_data["hand_2d"]

    meta = trajectory_data["metadata"]
    viz_width = int(meta["video_width"])
    viz_height = int(meta["video_height"])
    T = int(meta["num_frames"])

    # Convert (T,2) array to list of (u,v) tuples or None
    def array_to_list(arr: np.ndarray) -> List[Optional[Tuple[int, int]]]:
        uv_list: List[Optional[Tuple[int, int]]] = []
        for i in range(len(arr)):
            if np.isnan(arr[i, 0]) or np.isnan(arr[i, 1]):
                uv_list.append(None)
            else:
                uv_list.append((int(arr[i, 0]), int(arr[i, 1])))
        return uv_list

    hand_uv_2d = array_to_list(hand_2d)

    # English note: Prefer preserving the original FPS if available in metadata.
    fps = float(meta.get("fps", 30.0)) if isinstance(meta, dict) else 30.0
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 30.0

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (viz_width, viz_height))

    print(f"[INFO] Generating trajectory video: {out_p}")

    for i in range(T):
        if i < len(video_frames):
            frame = video_frames[i].copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # English note: Resize only when input frames differ from target size.
            if frame.shape[1] != viz_width or frame.shape[0] != viz_height:
                frame = cv2.resize(frame, (viz_width, viz_height))


            uv = hand_uv_2d[i]

            if output_path is not None:
                save_frame_log(
                    log_dir=out_p.parent / "trace_json",
                    frame_idx=i,
                    uv=uv,
                    fps=fps,
                )
        else:
            frame = np.ones((viz_height, viz_width, 3), np.uint8) * 255

        # Draw hand trajectory (single track)
        draw_sliding_trajectory(
            frame,
            hand_uv_2d,
            i,
            trace_window,
            line_color=(0, 0, 0),
            point_color=(0, 255, 0),  # green
            point_radius=8,
            line_thickness=4,
        )

        cv2.putText(
            frame,
            f"Frame {i+1}/{T}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        writer.write(frame)

    writer.release()
    print(f"[INFO] Saved trajectory video: {out_p}")


# ============================================================
# Convenience wrapper
# ============================================================
def generate_camera_trajectory_video(
    video_frames: np.ndarray,
    metadata: Dict[str, Any],
    output_path: str,
    trace_window: int = 60,
    viz_width: Optional[int] = None,
    viz_height: Optional[int] = None,
):
    """
    Generate 2D trajectory visualization video (projection + rendering).

    Args:
        video_frames: RGB frames (T, H, W, 3)
        metadata: Metadata dict
        output_path: Output video path
        trace_window: Sliding window length for visualization
        viz_width: Output width (defaults to input frame width)
        viz_height: Output height (defaults to input frame height)

    Returns:
        trajectory_data dict returned by project_camera_trajectories_to_2d()
    """
    if viz_width is None:
        viz_width = int(video_frames.shape[2])
    if viz_height is None:
        viz_height = int(video_frames.shape[1])

    trajectory_data = project_camera_trajectories_to_2d(
        metadata=metadata,
        num_frames=len(video_frames),
        viz_width=viz_width,
        viz_height=viz_height,
    )
    if trajectory_data is None:
        return None

    # English note: Record FPS if available from caller-side metadata.
    if "metadata" in trajectory_data and isinstance(trajectory_data["metadata"], dict):
        if "fps" not in trajectory_data["metadata"] and "fps" in metadata:
            trajectory_data["metadata"]["fps"] = metadata["fps"]

    generate_trajectory_visualization_video(
        video_frames=video_frames,
        trajectory_data=trajectory_data,
        output_path=output_path,
        trace_window=trace_window,
    )
    return trajectory_data

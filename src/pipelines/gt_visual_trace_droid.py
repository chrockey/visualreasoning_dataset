from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import h5py
import numpy as np
import yaml
import pyzed.sl as sl
import json

# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class Cfg:
    base_dir: Path
    out_dir: Path
    target_mp4_ids: List[str]
    eye: str
    rot_mode: str
    trace_window: int
    arm_cam_id: str
    gripper_offset_id: str


def load_cfg(path: str | Path) -> Cfg:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Cfg(
        base_dir=Path(raw["base_dir"]),
        out_dir=Path(raw["out_dir"]),
        target_mp4_ids=[str(x) for x in raw.get("target_mp4_ids", [])],
        eye=str(raw.get("eye", "left")),
        rot_mode=str(raw.get("rot_mode", "euler_xyz")),
        trace_window=int(raw.get("trace_window", 60)),
        arm_cam_id=str(raw["arm_cam_id"]),
        gripper_offset_id=str(raw["gripper_offset_id"]),
    )


# ============================================================
# Episode discovery
# ============================================================
def find_episode_dirs(base_dir: Path) -> List[Path]:
    """
    Find episode directories under base_dir by locating 'trajectory.h5'.
    Returns unique parent directories that contain trajectory.h5.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    traj_files = sorted(base_dir.rglob("trajectory.h5"))
    episode_dirs = sorted({p.parent for p in traj_files})
    return episode_dirs


def resolve_episode_assets(ep_dir: Path, mp4_id: str) -> tuple[Path, Path, Path]:
    """
    Resolve required assets for a given episode and mp4_id:
      - trajectory.h5
      - recordings/MP4/{mp4_id}.mp4
      - recordings/SVO/{mp4_id}.svo
    """
    traj_h5 = ep_dir / "trajectory.h5"
    mp4_path = ep_dir / "recordings" / "MP4" / f"{mp4_id}.mp4"
    svo_path = ep_dir / "recordings" / "SVO" / f"{mp4_id}.svo"

    if not traj_h5.exists():
        raise FileNotFoundError(f"Missing: {traj_h5}")
    if not mp4_path.exists():
        raise FileNotFoundError(f"Missing: {mp4_path}")
    if not svo_path.exists():
        raise FileNotFoundError(f"Missing: {svo_path}")

    return traj_h5, mp4_path, svo_path


def derive_view_cam_id(mp4_id: str) -> str:
    """
    Dataset rule:
      view_cam_id = "{mp4_id}_left"
    """
    return f"{mp4_id}_left"


# ============================================================
# ZED intrinsics: SVO -> fx,fy,cx,cy only
# ============================================================
def get_intrinsic_from_svo(
    svo_path: Path,
    eye: str = "left",
    coordinate_system: sl.COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
) -> tuple[float, float, float, float]:
    """
    Open an SVO and return (fx, fy, cx, cy) for the specified eye.
    """
    eye = eye.lower().strip()
    if eye not in ("left", "right"):
        raise ValueError(f"Invalid eye: {eye} (expected 'left' or 'right')")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.coordinate_system = coordinate_system

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {err} (svo={svo_path})")

    cam_info = zed.get_camera_information()

    # Version-tolerant access to calibration parameters
    if hasattr(cam_info, "camera_configuration") and hasattr(cam_info.camera_configuration, "calibration_parameters"):
        calib = cam_info.camera_configuration.calibration_parameters
    elif hasattr(cam_info, "calibration_parameters"):
        calib = cam_info.calibration_parameters
    else:
        zed.close()
        raise AttributeError("Cannot find calibration_parameters in camera information object.")

    cam = calib.left_cam if eye == "left" else calib.right_cam
    fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)

    zed.close()
    return fx, fy, cx, cy


# ============================================================
# H5 pose loading
# ============================================================
def load_pose_seq(h5_path: Path, key: str) -> np.ndarray:
    """
    Load pose sequence of shape (T, 6) from H5.
    pose6 = [x, y, z, rx, ry, rz]
    """
    with h5py.File(str(h5_path), "r") as f:
        if key not in f:
            raise KeyError(f"[H5] key not found: {key}")
        arr = np.array(f[key], dtype=np.float64)

    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"[H5] expected (T,6) for {key}, got {arr.shape}")
    return arr


# ============================================================
# SE(3) + projection utilities
# ============================================================
def euler_xyz_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def pose6_to_T(pose6: np.ndarray, rot_mode: str) -> np.ndarray:
    if rot_mode != "euler_xyz":
        raise ValueError(f"Unsupported rot_mode: {rot_mode} (expected 'euler_xyz')")

    x, y, z, rx, ry, rz = map(float, pose6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_xyz_to_R(rx, ry, rz)
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T


def world_to_cam_point(Pw: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    return R.T @ (Pw - t)


def project_point(
    Pc: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    W: int, H: int
) -> Optional[Tuple[int, int]]:
    X, Y, Z = Pc
    if Z <= 1e-6:
        return None
    u = cx + fx * (X / Z)
    v = cy + fy * (Y / Z)
    if 0 <= u < W and 0 <= v < H:
        return int(u), int(v)
    return None


# ============================================================
# Drawing
# ============================================================
def draw_sliding_trajectory(frame: np.ndarray, uv_all: List[Optional[Tuple[int, int]]], idx: int, window: int) -> None:
    start = max(0, idx - window + 1)
    prev = None
    for k in range(start, idx + 1):
        uv = uv_all[k]
        if uv is None:
            prev = None
            continue
        if prev is not None:
            cv2.line(frame, prev, uv, (0, 255, 0), 3)
        prev = uv

    cur = uv_all[idx]
    if cur is not None:
        cv2.circle(frame, cur, 9, (0, 0, 0), -1)
        cv2.circle(frame, cur, 7, (0, 255, 0), -1)


# ============================================================
# Single video render
# ============================================================
def render_one_video(
    traj_h5: Path,
    mp4_path: Path,
    svo_path: Path,
    out_path: Path,
    view_cam_id: str,
    arm_cam_id: str,
    gripper_offset_id: str,
    eye: str,
    rot_mode: str,
    trace_window: int,
) -> None:
    # Intrinsics from the matched SVO
    fx, fy, cx, cy = get_intrinsic_from_svo(svo_path, eye=eye)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # H5 keys
    key_view = f"observation/camera_extrinsics/{view_cam_id}"
    key_arm = f"observation/camera_extrinsics/{arm_cam_id}"
    key_off = f"observation/camera_extrinsics/{gripper_offset_id}"

    view_pose = load_pose_seq(traj_h5, key_view)
    arm_pose = load_pose_seq(traj_h5, key_arm)
    off_pose = load_pose_seq(traj_h5, key_off)

    T = min(len(view_pose), len(arm_pose), len(off_pose), n_frames)
    view_pose = view_pose[:T]
    arm_pose = arm_pose[:T]
    off_pose = off_pose[:T]

    # T_world_gripper = T_world_armCam @ T_armCam_gripperOffset
    gripper_world = np.zeros((T, 3), dtype=np.float64)
    for i in range(T):
        T_w_arm = pose6_to_T(arm_pose[i], rot_mode)
        T_arm_grip = pose6_to_T(off_pose[i], rot_mode)
        T_w_grip = T_w_arm @ T_arm_grip
        gripper_world[i] = T_w_grip[:3, 3]

    # Project into the view camera
    uv_list: List[Optional[Tuple[int, int]]] = []
    for i in range(T):
        T_w_view = pose6_to_T(view_pose[i], rot_mode)
        Pc = world_to_cam_point(gripper_world[i], T_w_view)
        uv_list.append(project_point(Pc, fx, fy, cx, cy, W, H))
        out_json_path = out_path.with_suffix(".json")
    
    save_uv_json(
        out_json_path=out_json_path,
        uv_list=uv_list,
        W=W,
        H=H,
        fps=fps,
        eye=eye,
        trace_window=trace_window,
        video_name=mp4_path.name,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(T):
        ok, frame = cap.read()
        if not ok:
            break
        draw_sliding_trajectory(frame, uv_list, i, trace_window)
        writer.write(frame)

    cap.release()
    writer.release()

    print(f"[OK] {out_path}")


def save_uv_json(
    out_json_path: Path,
    uv_list: List[Optional[Tuple[int, int]]],
    W: int,
    H: int,
    fps: float,
    eye: str,
    trace_window: int,
    video_name: str,
):
    data = {
        "meta": {
            "video": video_name,
            "width": W,
            "height": H,
            "fps": fps,
            "eye": eye,
            "trace_window": trace_window,
        },
        "frames": []
    }

    for i, uv in enumerate(uv_list):
        if uv is None:
            data["frames"].append({
                "frame_idx": i,
                "u": None,
                "v": None,
                "visible": False
            })
        else:
            u, v = uv
            data["frames"].append({
                "frame_idx": i,
                "u": int(u),
                "v": int(v),
                "visible": True
            })

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


# ============================================================
# Batch runner (date dir -> episodes -> two mp4 ids)
# ============================================================
def run_batch(cfg: Cfg) -> None:
    episodes = find_episode_dirs(cfg.base_dir)
    if not episodes:
        raise RuntimeError(f"No episodes found under: {cfg.base_dir}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for ep_dir in episodes:
        ep_name = ep_dir.name

        for mp4_id in cfg.target_mp4_ids:
            # Skip stereo files implicitly by targeting only exact '{id}.mp4'
            try:
                traj_h5, mp4_path, svo_path = resolve_episode_assets(ep_dir, mp4_id)
            except FileNotFoundError:
                # Not every episode has every camera; skip quietly
                continue

            view_cam_id = derive_view_cam_id(mp4_id)

            # Output layout: out_dir/<episode_name>/<id>_viz_trace.mp4
            out_path = cfg.out_dir / ep_name / f"{mp4_id}_viz_trace.mp4"

            try:
                render_one_video(
                    traj_h5=traj_h5,
                    mp4_path=mp4_path,
                    svo_path=svo_path,
                    out_path=out_path,
                    view_cam_id=view_cam_id,
                    arm_cam_id=cfg.arm_cam_id,
                    gripper_offset_id=cfg.gripper_offset_id,
                    eye=cfg.eye,
                    rot_mode=cfg.rot_mode,
                    trace_window=cfg.trace_window,
                )
            except Exception as e:
                print(f"[WARN] failed ep={ep_name}, id={mp4_id}: {e}")


def main() -> None:
    cfg = load_cfg("./droid_gt_trace.yaml")
    run_batch(cfg)


if __name__ == "__main__":
    main()

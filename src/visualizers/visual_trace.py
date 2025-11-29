from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import torch
import os
from matplotlib import cm
from PIL import Image, ImageDraw
import imageio


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb

def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb

def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class VisualTraceVisualizer:
    """Visualization utilities for VisualTracePipeline."""

    def __init__(
        self,
        save_dir: str = "./results",
        fps: int = 30,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 1,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.tracks_leave_trace = tracks_leave_trace
        self.linewidth = linewidth
        self.fps = fps
        self._video_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def _generate_rainbow_palette(self, num_points: int) -> np.ndarray:
        if num_points <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        color_positions = np.linspace(0.0, 1.0, num_points, endpoint=False, dtype=np.float32)
        colors = np.array([self.color_map(pos % 1.0)[:3] for pos in color_positions], dtype=np.float32)
        return colors * 255.0

    def save_visualizations(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        keypoints: np.ndarray,
        prefix: str = "",
        video_name: str = "visual_trace",
        frame_idx: Optional[int] = None,
        finalize: bool = False,
    ):
        """Save single-frame visualizations and accumulate frames for video export."""
        os.makedirs(self.save_dir, exist_ok=True)
        base_path = os.path.join(self.save_dir, video_name)
        os.makedirs(base_path, exist_ok=True)
        frame_tag = f"{frame_idx:05d}_" if frame_idx is not None else ""
        prefix_tag = f"{prefix}_" if prefix else ""

        image_array = (
            image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
        )
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(
                f"Expected image of shape (H, W, 3), got {image_array.shape}"
            )
        image_array = np.ascontiguousarray(image_array)
        image_to_save = image_array
        if image_to_save.dtype != np.uint8:
            image_to_save = np.clip(image_to_save, 0, 255).astype(np.uint8)

        # Save the raw input frame for quick inspection
        Image.fromarray(image_to_save).save(
            os.path.join(base_path, f"{frame_tag}{prefix_tag}pre_image.png")
        )

        # Prepare tensors for visualization helpers
        video_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        keypoints_array = (
            keypoints.detach().cpu().numpy()
            if isinstance(keypoints, torch.Tensor)
            else np.asarray(keypoints)
        )
        keypoints_array = np.ascontiguousarray(keypoints_array)
        if keypoints_array.ndim != 2 or keypoints_array.shape[-1] != 2:
            raise ValueError(
                f"Expected keypoints of shape (N, 2), got {keypoints_array.shape}"
            )
        tracks_tensor = torch.from_numpy(keypoints_array).unsqueeze(0).unsqueeze(0).float()

        segm_vector = None
        if mask is not None:
            mask_array = (
                mask.detach().cpu().numpy()
                if isinstance(mask, torch.Tensor)
                else np.asarray(mask)
            )
            mask_array = np.ascontiguousarray(mask_array)
            if mask_array.ndim != 2:
                raise ValueError(
                    f"Expected mask of shape (H, W), got {mask_array.shape}"
                )
            coords = np.round(keypoints_array).astype(int)
            coords[:, 0] = np.clip(coords[:, 0], 0, mask_array.shape[1] - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, mask_array.shape[0] - 1)
            segm_values = mask_array[coords[:, 1], coords[:, 0]].astype(np.int64)
            if not np.any(segm_values > 0):
                error_msg = (
                    "[VisualTraceVisualizer] No keypoints fall inside positive "
                    "segmentation regions; aborting visualization."
                )
                print(error_msg)
                # raise RuntimeError(error_msg)
            else:
                segm_vector = torch.from_numpy(segm_values)

        color_alpha = 255

        rendered = self.draw_tracks_on_video(
            video=video_tensor,
            tracks=tracks_tensor,
            segm_mask=segm_vector,
            query_frame=0,
            color_alpha=color_alpha,
            tracks_leave_trace_override=0,
        )
        rendered = rendered.detach().cpu()
        annotated_frame = rendered[0, 0].permute(1, 2, 0).numpy().astype(np.uint8)
        Image.fromarray(annotated_frame).save(
            os.path.join(base_path, f"{frame_tag}{prefix_tag}keypoints.png")
        )

        frame_tensor = (
            torch.from_numpy(annotated_frame)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(torch.uint8)
        )
        self._video_buffers[video_name].append(frame_tensor)
        if finalize and self._video_buffers[video_name]:
            full_video = torch.cat(self._video_buffers[video_name], dim=1)
            self.save_video(full_video, filename=video_name)
            self._video_buffers.pop(video_name, None)

    def save_trace_video(
        self,
        frames: np.ndarray,
        tracks: np.ndarray,
        video_name: str,
        mask: Optional[np.ndarray] = None,
        start_frame_idx: int = 0,
        end_frame_idx: Optional[int] = None,
        leave_trace: bool = True,
    ):
        frames_array = (
            frames.detach().cpu().numpy() if isinstance(frames, torch.Tensor) else np.asarray(frames)
        )
        if frames_array.ndim != 4 or frames_array.shape[-1] != 3:
            raise ValueError(
                f"Expected frames of shape (T, H, W, 3), got {frames_array.shape}"
            )
        clip_start = max(0, start_frame_idx)
        clip_end = frames_array.shape[0] if end_frame_idx is None else min(
            frames_array.shape[0], end_frame_idx + 1
        )
        if clip_end <= clip_start:
            raise ValueError(
                f"Invalid frame range [{clip_start}, {clip_end}) for frames of length {frames_array.shape[0]}"
            )
        frames_array = frames_array[clip_start:clip_end]
        if frames_array.dtype != np.uint8:
            frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)

        tracks_array = (
            tracks.detach().cpu().numpy() if isinstance(tracks, torch.Tensor) else np.asarray(tracks)
        )
        if tracks_array.ndim != 3 or tracks_array.shape[-1] != 2:
            raise ValueError(
                f"Expected tracks of shape (T, N, 2), got {tracks_array.shape}"
            )
        tracks_array = tracks_array[clip_start:clip_end]

        video_tensor = (
            torch.from_numpy(frames_array)
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
        )
        tracks_tensor = torch.from_numpy(tracks_array).unsqueeze(0).float()

        segm_vector = None
        if mask is not None:
            mask_array = (
                mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
            )
            if mask_array.ndim != 2:
                raise ValueError(
                    f"Expected mask of shape (H, W), got {mask_array.shape}"
                )
            first_coords = np.round(tracks_array[0]).astype(int)
            first_coords[:, 0] = np.clip(first_coords[:, 0], 0, mask_array.shape[1] - 1)
            first_coords[:, 1] = np.clip(first_coords[:, 1], 0, mask_array.shape[0] - 1)
            segm_values = mask_array[first_coords[:, 1], first_coords[:, 0]].astype(np.int64)
            segm_vector = torch.from_numpy(segm_values)

        rendered = self.draw_tracks_on_video(
            video=video_tensor,
            tracks=tracks_tensor,
            segm_mask=segm_vector,
            query_frame=0,
            color_alpha=255,
            tracks_leave_trace_override=(
                self.tracks_leave_trace if leave_trace else 0
            ),
        )
        rendered = rendered.detach().cpu().to(torch.uint8)
        self.save_video(rendered, filename=video_name)

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in video.unbind(1)]

            # Prepare the video file path (store next to frame directory)
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")

            # Create a writer object
            video_writer = imageio.get_writer(save_path, fps=self.fps)

            # Write frames to the video file
            for frame in wide_list:
                video_writer.append_data(frame)

            video_writer.close()

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        query_frame=0,
        compensate_for_camera_motion=False,
        color_alpha: int = 255,
        tracks_leave_trace_override: Optional[int] = None,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if segm_mask is not None:
            if torch.is_tensor(segm_mask):
                segm_mask = segm_mask.detach().cpu().numpy()
            else:
                segm_mask = np.asarray(segm_mask)

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        vector_colors = np.zeros((T, N, 3), dtype=np.float32)

        if self.mode == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif self.mode == "rainbow":
            palette = self._generate_rainbow_palette(N)
            if palette.shape[0] > 0:
                vector_colors = np.repeat(palette[None], T, axis=0)
        elif segm_mask is None:
            # color changes with time for other modes
            for t in range(T):
                color = np.array(self.color_map(t / max(T, 1))[:3])[None] * 255
                vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            # color changes with segm class for non-rainbow modes
            color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
            color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
            color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
            vector_colors = np.repeat(color[None], T, axis=0)

        tracks_leave_trace = (
            self.tracks_leave_trace
            if tracks_leave_trace_override is None
            else tracks_leave_trace_override
        )

        #  draw tracks
        if tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - tracks_leave_trace)
                    if tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )

        #  draw points
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                is_visible = True
                if visibility is not None:
                    vis_value = visibility[0, t, i]
                    if torch.is_tensor(vis_value):
                        is_visible = bool(vis_value.item())
                    elif isinstance(vis_value, np.ndarray):
                        is_visible = bool(vis_value.reshape(-1)[0])
                    else:
                        is_visible = bool(vis_value)
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion
                        and segm_mask is not None
                        and segm_mask[i] > 0
                    ):
                        img = draw_circle(
                            img,
                            coord=coord,
                            radius=int(self.linewidth * 2),
                            color=vector_colors[t, i].astype(int),
                            visible=is_visible,
                            color_alpha=color_alpha,
                        )
            res_video[t] = np.array(img)

        #  construct the final rgb sequence
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].astype(int),
                        self.linewidth,
                    )
            if self.tracks_leave_trace > 0:
                rgb = Image.fromarray(
                    np.uint8(
                        add_weighted(
                            np.array(rgb), alpha, np.array(original), 1 - alpha, 0
                        )
                    )
                )
        rgb = np.array(rgb)
        return rgb

from typing import Literal, Optional
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


class SAM2:
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-small",
        mask_selection_mode: Literal[
            "highest_score", "smallest_mask", "random"
        ] = "smallest_mask",
    ):
        self.model_id = model_id
        self.sam = SAM2ImagePredictor.from_pretrained(
            self.model_id,
            hydra_overrides_extra=["++model.compile_image_encoder=True"],
        )
        assert mask_selection_mode in [
            "highest_score",
            "smallest_mask",
            "random",
        ], "Invalid mask selection mode"
        self.mask_selection_mode = mask_selection_mode

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def inference(self, image, points):
        if self.sam is None:
            raise ValueError("SAM model is not initialized")

        self.sam.set_image(image)
        point_coords = points[:, np.newaxis, :]
        point_labels = np.ones(len(points))[:, np.newaxis]
        logits, scores, *_ = self.sam.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=True,
        )
        masks = logits > 0
        return masks, scores, logits

    def __call__(self, image: np.ndarray, points: np.ndarray):
        # masks.shape (num_queries, 3, H, W)
        masks, scores, logits = self.inference(image, points)
        if points.shape[0] == 1:
            masks = masks[np.newaxis, ...]
            scores = scores[np.newaxis, ...]
            logits = logits[np.newaxis, ...]
        arange = np.arange(len(masks))
        if self.mask_selection_mode == "highest_score":
            argmax = np.argmax(scores, axis=-1)
            mask = masks[arange, argmax]
            score = scores[arange, argmax]
            logit = logits[arange, argmax]
        elif self.mask_selection_mode == "smallest_mask":
            mask_area = masks.sum(axis=(-2, -1))
            argmin = np.argmin(mask_area, axis=-1)
            mask = masks[arange, argmin]
            score = scores[arange, argmin]
            logit = logits[arange, argmin]
        elif self.mask_selection_mode == "random":
            idx = np.random.choice(masks.shape[0])
            mask = masks[arange, idx]
            score = scores[arange, idx]
            logit = logits[arange, idx]
        return mask, score, logit


class SAM2VideoPredictorWrapper:
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-small",
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        **kwargs,
    ):
        """Initialize SAM2 Video Predictor wrapper.
        
        Args:
            model_id: HuggingFace model ID for SAM2
            offload_video_to_cpu: Whether to offload video frames to CPU memory
            offload_state_to_cpu: Whether to offload inference state to CPU memory
            **kwargs: Additional arguments passed to from_pretrained
        """
        self.model_id = model_id
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self.base = SAM2VideoPredictor.from_pretrained(
            self.model_id,
            hydra_overrides_extra=["++model.compile_image_encoder=True"],
            **kwargs,
        )

    def _preprocess_frames(
        self,
        frames: np.ndarray,
        offload_video_to_cpu: bool,
        img_mean: tuple = (0.485, 0.456, 0.406),
        img_std: tuple = (0.229, 0.224, 0.225),
    ) -> tuple[torch.Tensor, int, int]:
        """Preprocess video frames for SAM2.
        
        Args:
            frames: Array of shape (T, H, W, 3) - video frames
            offload_video_to_cpu: Whether to keep frames on CPU
            img_mean: Image normalization mean
            img_std: Image normalization std
            
        Returns:
            Tuple of (preprocessed_images, video_height, video_width)
        """
        T, H, W, C = frames.shape
        video_height, video_width = H, W
        
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        
        images = []
        for i in range(T):
            frame = frames[i]
            frame = Image.fromarray(frame.astype("uint8"))
            frame = frame.resize((self.base.image_size, self.base.image_size), Image.BILINEAR)
            frame = np.array(frame)
            # Convert (H,W,C) → (C,H,W)
            tensor = torch.tensor(frame).permute(2, 0, 1)  # (3, H, W)
            images.append(tensor)

        # stack to (T, 3, H, W)
        images = torch.stack(images).float() / 255.0

        # move to GPU if needed
        compute_device = self.base.device
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)

        # normalize
        images = (images - img_mean) / img_std
        return images, video_height, video_width

    @torch.inference_mode()
    def _init_state(
        self,
        frames: np.ndarray,
        offload_video_to_cpu: Optional[bool] = None,
        offload_state_to_cpu: Optional[bool] = None,
    ) -> dict:
        """Initialize an inference state for video processing.
        
        Args:
            frames: Array of shape (T, H, W, 3) - video frames
            offload_video_to_cpu: Whether to offload video frames to CPU (uses instance default if None)
            offload_state_to_cpu: Whether to offload state to CPU (uses instance default if None)
            
        Returns:
            Inference state dictionary
        """
        if offload_video_to_cpu is None:
            offload_video_to_cpu = self.offload_video_to_cpu
        if offload_state_to_cpu is None:
            offload_state_to_cpu = self.offload_state_to_cpu
        
        images, video_height, video_width = self._preprocess_frames(
            frames, offload_video_to_cpu
        )
        
        compute_device = self.base.device
        inference_state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": compute_device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        
        # Warm up the visual backbone and cache the image feature on frame 0
        self.base._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state
    
    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def video_inference(
        self,
        video_frames: np.ndarray,
        points: Optional = None,
        points_frame_idx: int = 0,
        reverse: bool = False,
        mask: Optional = None,
    ) -> dict[int, np.ndarray]:
        """Video segmentation using SAM2 video predictor.
        
        Args:
            video_frames: Array of shape (T, H, W, 3) - video frames
            points: Optional array of shape (num_points, 2) - point coordinates
            mask: Optional binary mask as numpy array (H, W) - initial mask
            points_frame_idx: Index of the frame to add prompts (default: 0)
            reverse: Whether to use reverse propagation (default: False)
        
        Returns:
            Dictionary mapping frame indices to binary masks
        
        Raises:
            ValueError: If neither points nor mask is provided, or both are provided
        """
        # Validate inputs
        if points is None and mask is None:
            raise ValueError("Either points or mask must be provided")
        if points is not None and mask is not None:
            raise ValueError("Cannot provide both points and mask, choose one")
        
        # Initialize video state
        state = self._init_state(video_frames)
        
        # Add prompt based on input type
        if mask is None:
            # Points input path
            if points is None or len(points) == 0:
                return {}
            
            point_labels = np.ones(len(points))
            self.base.add_new_points_or_box(
                state,
                frame_idx=points_frame_idx,
                obj_id=1,
                points=points,
                labels=point_labels,
            )
        else:
            # Mask input path
            if mask.sum() == 0:
                return {}
            
            self.base.add_new_mask(
                state,
                frame_idx=points_frame_idx,
                obj_id=1,
                mask=mask,
            )
                    
        # Propagate through video with reverse option
        video_segments = {}
        for frame_idx, object_ids, masks in self.base.propagate_in_video(state, reverse=reverse):
            video_segments[frame_idx] = (masks[0] > 0.0).cpu().numpy()
        
        return video_segments

    def video_inference_with_mask(
        self,
        video_frames: np.ndarray,
        mask: np.ndarray,
        mask_frame_idx: int = 0,
        reverse: bool = False,
    ) -> dict[int, np.ndarray]:
        """Video segmentation using SAM2 video predictor with mask input.
        
        Convenience wrapper for video_inference with mask input.
        
        Args:
            video_frames: Array of shape (T, H, W, 3) - video frames
            mask: Binary mask as numpy array (H, W) - initial mask
            mask_frame_idx: Index of the frame where mask is located (default: 0)
            reverse: Whether to use reverse propagation (default: False)
        
        Returns:
            Dictionary mapping frame indices to binary masks
        """
        
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).float().to(self.base.device)
        else:
            mask_tensor = mask
            
        return self.video_inference(
            video_frames=video_frames,
            mask=mask,
            points_frame_idx=mask_frame_idx,
            reverse=reverse,
        )

    def process_video(
        self,
        video_frames: np.ndarray,
        points: np.ndarray,
        points_frame_index: int,
        reverse: bool = False,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> dict[int, np.ndarray]:
        """Process video frames using points and propagation.
        
        Args:
            video_frames: Array of shape (T, H, W, 3) - input video frames
            points: Array of shape (num_points, 2) - point coordinates
            points_frame_index: Frame index where points are located
            reverse: Whether to use reverse propagation
            start_frame: Start frame for propagation range (required if reverse=True)
            end_frame: End frame for propagation range (required if reverse=True)
        
        Returns:
            Dictionary mapping frame indices to binary masks
        """
        if points is None or len(points) == 0:
            return {}
            
        if reverse:
            if start_frame is None or end_frame is None:
                raise ValueError("start_frame and end_frame must be provided when reverse=True")
            
            # Extract segment frames for reverse propagation
            segment_frames = video_frames[start_frame : end_frame + 1]
            adjusted_points_frame_idx = points_frame_index - start_frame
            
            # Run video inference on segment with reverse=True
            segment_results = self.video_inference(
                segment_frames, points, adjusted_points_frame_idx, reverse=True
            )
            
            # Map results back to original frame indices
            final_results = {}
            for seg_idx, mask in segment_results.items():
                original_idx = start_frame + seg_idx
                final_results[original_idx] = mask
            
            return final_results
        else:
            # Normal mode: propagate forward from points_frame_index
            return self.video_inference(video_frames, points, points_frame_index, reverse=False)


if __name__ == "__main__":
    import cv2

    images = [
        cv2.imread(f"demo/16341/campos_512_v4/{i:05d}/{i:05d}.png")
        for i in range(1, 5)
    ]
    model = SAM2()
    for image in images:
        masks, scores, logits = model(image)
        print(masks)
        print(scores)
        print(logits)

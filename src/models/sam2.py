from typing import Literal, Optional
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from PIL import Image

class SAM2VideoPredictorWrapper():
    def __init__(self, model_id, **kwargs):
        self.base = SAM2VideoPredictor.from_pretrained(model_id, **kwargs)

    def preprocess_frames(
        self,
        frames,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
    ):
        import torch
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
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)

        # normalize
        images = (images - img_mean) / img_std
        return images, video_height, video_width

    @torch.inference_mode()
    def init_state(
        self,
        frames,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False
    ):
        """Initialize an inference state."""
        from collections import OrderedDict
        
        images, video_height, video_width = self.preprocess_frames(
            frames, offload_video_to_cpu
        )
        
        compute_device = self.base.device  # device of the model
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self.base._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state



class SAM2:
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-small",
        mask_selection_mode: Literal[
            "highest_score", "smallest_mask", "random"
        ] = "smallest_mask",
        mode: Literal["image", "video"] = "image",
        checkpoint_path: Optional[str] = None,
        model_cfg: Optional[str] = None,
    ):
        self.model_id = model_id
        self.mask_selection_mode = mask_selection_mode
        self.mode = mode

        assert mask_selection_mode in [
            "highest_score",
            "smallest_mask", 
            "random",
        ], "Invalid mask selection mode"
        
        if self.mode  == "image":
            self.sam = SAM2ImagePredictor.from_pretrained(
                self.model_id,
                hydra_overrides_extra=["++model.compile_image_encoder=True"],
            )
        elif self.mode  == "video":
            self.sam_wrapper= SAM2VideoPredictorWrapper(
                self.model_id,
                hydra_overrides_extra=["++model.compile_image_encoder=True"]
            )

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def video_inference(self, video_frames: np.ndarray, points: np.ndarray, points_frame_idx: int = 0, reverse: bool = False):
        """Video segmentation using SAM2 video predictor.
        
        Args:
            video_frames: Array of shape (N, H, W, 3) - video frames
            points: Array of shape (num_points, 2) - points
            points_frame_idx: Index of the frame to add prompts (default: 0)
            reverse: Whether to use reverse propagation (default: False)
        
        Returns:
            Dictionary of masks for each frame
        """
        # Initialize video state
        state = self.sam_wrapper.init_state(video_frames)
        
        # Add points to the specified frame
        point_coords = points
        point_labels = np.ones(len(points))
        
        # Add new points
        frame_idx, object_ids, masks = self.sam_wrapper.base.add_new_points_or_box(
            state,
            frame_idx=points_frame_idx,
            obj_id=1,  # Single object
            points=point_coords,
            labels=point_labels,
        )
        # Propagate through video with reverse option
        video_segments = {}
        for frame_idx, object_ids, masks in self.sam_wrapper.base.propagate_in_video(state, reverse=reverse):
            video_segments[frame_idx] = (masks[0] > 0.0).cpu().numpy()
        return video_segments


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

    def __call__(self, frames, points, points_frame_index: Optional[int] = None, reverse: bool = False, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Unified interface for both image and video segmentation.
        
        Args:
            frames: Input frames
            points: Input points
            points_frame_index: Frame index where points are located
            reverse: Whether to use reverse propagation (backward from points_frame_index)
            start_frame: Start frame for propagation (only used with reverse=True)
            end_frame: End frame for propagation (only used with reverse=True)
        """
        if self.mode == "image":
            # Single image mode - original functionality
            return self._process_single_image(frames, points)
        elif self.mode == "video":
            # Video mode - use points and propagate
            return self._process_video(frames, points, points_frame_index, reverse, start_frame, end_frame)
    
    def _process_single_image(self, image: np.ndarray, points: np.ndarray):
        """Process single image - original functionality."""
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
    
    def _process_video(self, video_frames: np.ndarray, points: np.ndarray, points_frame_index, reverse: bool = False, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Process video frames using points and propagation.
        
        Args:
            video_frames: Input video frames
            points: Input points
            points_frame_index: Frame index where points are located
            reverse: Whether to use reverse propagation
            start_frame: Start frame for propagation range
            end_frame: End frame for propagation range
        """
        if points is None or len(points) == 0:
            return {}
            
        if reverse:
            # Extract segment frames for reverse propagation
            segment_frames = video_frames[start_frame:end_frame+1]
            adjusted_points_frame_idx = points_frame_index - start_frame
            
            # Run video inference on segment with reverse=True
            segment_results = self.video_inference(segment_frames, points, adjusted_points_frame_idx, reverse=True)
            
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

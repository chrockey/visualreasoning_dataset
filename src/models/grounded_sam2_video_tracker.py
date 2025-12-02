import os
import torch
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Generator, List
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor
from src.models.grounded_sam2 import GroundedSAM2
from src.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo


class GroundedSAM2VideoTracker:
    """
    Combines Grounding DINO detection with SAM2 video tracking.
    Handles continuous ID tracking across video frames using sliding window approach.

    This class wraps:
    - GroundedSAM2: For initial object detection on keyframes
    - SAM2 video predictor: For temporal mask propagation
    - MaskDictionaryModel: For continuous object ID tracking
    """

    def __init__(
        self,
        grounded_sam2: GroundedSAM2,
        sam2_checkpoint: str = None,
        model_cfg: str = None,
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda"
    ):
        """
        Initialize the video tracker.

        Args:
            grounded_sam2: Existing GroundedSAM2 instance for detection
            sam2_checkpoint: Optional path to SAM2 checkpoint file (for local checkpoints)
            model_cfg: Optional path to SAM2 model config file (for local checkpoints)
            model_id: HuggingFace model ID (used if checkpoint/config not provided)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.grounded_sam2 = grounded_sam2
        self.device = device

        # Build SAM2 video predictor
        # If checkpoint and config are provided, use them (local checkpoint mode)
        # Otherwise, use HuggingFace model (default)
        if sam2_checkpoint is not None and model_cfg is not None:
            print(f"Loading SAM2 video predictor from checkpoint: {sam2_checkpoint}")
            self.video_predictor = build_sam2_video_predictor(
                model_cfg,
                sam2_checkpoint,
                device=device
            )
        else:
            print(f"Loading SAM2 video predictor from HuggingFace: {model_id}")
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            self.video_predictor = SAM2VideoPredictor.from_pretrained(
                model_id,
                device=device
            )

        # SAM2 uses bfloat16 by default on CUDA for efficiency
        # Store the model dtype for later use
        if hasattr(self.video_predictor, 'model'):
            # Get the dtype of the first parameter to know what dtype SAM2 is using
            first_param = next(self.video_predictor.model.parameters())
            self.model_dtype = first_param.dtype
            print(f"SAM2 video predictor using dtype: {self.model_dtype}")
        else:
            self.model_dtype = torch.float32

    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = True,
        async_loading_frames: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize video predictor state from a directory of frames.

        Args:
            video_path: Path to directory containing video frames
            offload_video_to_cpu: Whether to offload frames to CPU memory
            async_loading_frames: Whether to load frames asynchronously

        Returns:
            Inference state dictionary
        """
        inference_state = self.video_predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames
        )
        return inference_state

    def reset_state(self, inference_state: Dict[str, Any]) -> None:
        """
        Reset the video predictor state.

        Args:
            inference_state: Inference state dictionary to reset
        """
        self.video_predictor.reset_state(inference_state)

    def save_frames_to_directory(self, frames: np.ndarray, output_dir: str) -> List[str]:
        """
        Save video frames to a temporary directory for SAM2 video predictor.

        Args:
            frames: Video frames (N, H, W, 3) in RGB format
            output_dir: Directory to save frames

        Returns:
            List of frame filenames (sorted)
        """
        os.makedirs(output_dir, exist_ok=True)
        frame_names = []

        for i, frame in enumerate(frames):
            frame_name = f"{i:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
            frame_names.append(frame_name)

        return frame_names

    def add_new_mask(
        self,
        inference_state: Dict[str, Any],
        frame_idx: int,
        obj_id: int,
        mask: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Add a new mask to the video predictor at a specific frame.

        Args:
            inference_state: Inference state dictionary
            frame_idx: Frame index to add mask
            obj_id: Object ID for this mask
            mask: Binary mask tensor (H, W)

        Returns:
            Tuple of (frame_idx, out_obj_ids, out_mask_logits)
        """
        frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
            inference_state,
            frame_idx,
            obj_id,
            mask
        )
        return frame_idx, out_obj_ids, out_mask_logits

    def propagate_in_video(
        self,
        inference_state: Dict[str, Any],
        start_frame_idx: int = 0,
        max_frame_num_to_track: int = None,
        reverse: bool = False
    ) -> Generator[Tuple[int, torch.Tensor, torch.Tensor], None, None]:
        """
        Propagate masks through video frames.

        Args:
            inference_state: Inference state dictionary
            start_frame_idx: Starting frame index for propagation
            max_frame_num_to_track: Maximum number of frames to track
            reverse: Whether to propagate in reverse (backward in time)

        Yields:
            Tuple of (frame_idx, out_obj_ids, out_mask_logits) for each frame
        """
        for frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse
        ):
            yield frame_idx, out_obj_ids, out_mask_logits

    def propagate_object_bidirectional(
        self,
        inference_state: Dict[str, Any],
        anchor_frame: int,
        anchor_mask: torch.Tensor,
        obj_id: int,
        class_name: str,
        start_frame: int,
        end_frame: int,
        all_frame_masks: Dict[int, Dict[int, Dict]],
        reverse: bool = False
    ):
        """
        Propagate a single object's mask bidirectionally using SAM2.

        Args:
            inference_state: SAM2 video predictor state
            anchor_frame: Frame index where object is detected
            anchor_mask: Mask of the object at anchor frame
            obj_id: Object ID
            class_name: Object class name
            start_frame: Start frame for propagation
            end_frame: End frame for propagation
            all_frame_masks: Dictionary to store results
            reverse: Whether to propagate in reverse (backward)
        """
        # Reset state for this propagation
        self.reset_state(inference_state)

        # Ensure mask has correct dtype and is on correct device
        # Match the model's dtype (bfloat16 on CUDA, float32 on CPU)
        if anchor_mask.dtype != self.model_dtype:
            anchor_mask = anchor_mask.to(self.model_dtype)
        if str(anchor_mask.device) != str(self.device):
            anchor_mask = anchor_mask.to(self.device)

        # Add the anchor mask
        self.add_new_mask(
            inference_state,
            anchor_frame,
            obj_id,
            anchor_mask
        )

        # Propagate through frames using SAM2's native reverse support
        # When reverse=True, SAM2 propagates backward in time from anchor_frame
        # When reverse=False, SAM2 propagates forward in time from anchor_frame
        num_frames = abs(end_frame - start_frame) + 1

        for out_frame_idx, out_obj_ids, out_mask_logits in self.propagate_in_video(
            inference_state,
            start_frame_idx=anchor_frame,
            max_frame_num_to_track=num_frames,
            reverse=reverse
        ):
            # Only store frames in the requested range
            if start_frame <= out_frame_idx <= end_frame:
                for i, out_obj_id in enumerate(out_obj_ids):
                    if out_obj_id == obj_id:
                        out_mask = (out_mask_logits[i] > 0.0)[0]

                        # Store mask
                        all_frame_masks[out_frame_idx][obj_id] = {
                            'mask': out_mask,
                            'class_name': class_name,
                            'mask_size': out_mask.sum().item()
                        }
                        break

    def detect_and_track(
        self,
        frames: np.ndarray,
        text_prompt: str,
        step: int = 20,
        iou_threshold: float = 0.8
    ) -> Dict[int, MaskDictionaryModel]:
        """
        High-level interface: detect objects and track them across video.

        This implements the sliding window approach from the reference code:
        1. Sample keyframes every `step` frames
        2. Run Grounding DINO + SAM2 on keyframes
        3. Track continuous IDs using IoU matching
        4. Propagate masks through intermediate frames

        Args:
            frames: Video frames as numpy array (N, H, W, 3)
            text_prompt: Text prompt for Grounding DINO
            step: Keyframe sampling interval
            iou_threshold: IoU threshold for object matching

        Returns:
            Dictionary mapping frame_idx -> MaskDictionaryModel
        """
        # This would need temporary directory creation for frames
        # We'll implement this in the pipeline method instead
        # to keep this class focused on the core tracking logic
        raise NotImplementedError(
            "Use the lower-level methods (init_state, add_new_mask, propagate_in_video) "
            "for more control. See _process_video_mode_with_continuous_id() for usage example."
        )

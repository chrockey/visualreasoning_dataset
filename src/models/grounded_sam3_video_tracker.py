import os
import gc
import torch
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Generator, List, Optional
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor
from src.models.grounded_sam2 import GroundedSAM2
from src.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo


class GroundedSAM3VideoTracker:
    """
    Combines Grounding DINO detection with SAM3 video tracking.
    Handles continuous ID tracking across video frames using text prompts.

    This class wraps:
    - GroundedSAM2: For initial object detection on keyframes
    - SAM3 video predictor: For temporal mask propagation with text prompts
    - MaskDictionaryModel: For continuous object ID tracking

    Note: SAM3 uses a session-based API and supports text-based detection,
    which differs from SAM2's point-based approach.
    """

    def __init__(
        self,
        grounded_sam2: GroundedSAM2,
        gpus_to_use: Optional[List[int]] = None,
        device: str = "cuda",
        model_id: str = "facebook/sam3"
    ):
        """
        Initialize the SAM3 video tracker.

        Args:
            grounded_sam2: Existing GroundedSAM2 instance for detection
            gpus_to_use: List of GPU indices to use for SAM3 (default: all available GPUs)
            device: Device to run on ('cuda' or 'cpu')
            model_id: Hugging Face model ID for SAM3 (default: facebook/sam3)
        """
        self.grounded_sam2 = grounded_sam2
        self.device = device
        self.model_id = model_id

        # Build SAM3 video predictor
        if gpus_to_use is None:
            gpus_to_use = range(torch.cuda.device_count()) if torch.cuda.is_available() else [0]

        print(f"Loading SAM3 video predictor from {model_id} with GPUs: {list(gpus_to_use)}")
        self.video_predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

        # SAM3 uses bfloat16 by default on CUDA for efficiency
        self.model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"SAM3 video predictor using dtype: {self.model_dtype}")

        # Track active session
        self.session_id = None
        self.video_path = None

    def init_state(
        self,
        video_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initialize video predictor state from a directory of frames.

        Args:
            video_path: Path to directory containing video frames or MP4 file
            **kwargs: Additional arguments (ignored for compatibility with SAM2)

        Returns:
            Inference state dictionary containing session_id and actual SAM3 state
        """
        # Start a new session with SAM3
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )

        self.session_id = response["session_id"]
        self.video_path = video_path

        # Get the actual SAM3 inference state from the session
        session = self.video_predictor._ALL_INFERENCE_STATES[self.session_id]
        sam3_state = session["state"]

        # Return state dictionary compatible with SAM2 API
        # Include both session_id for API calls and actual state for direct model access
        inference_state = {
            "session_id": self.session_id,
            "video_path": video_path,
            "predictor": self.video_predictor,
            "state": sam3_state  # Actual SAM3 inference state for direct tracker access
        }

        print(f"SAM3 session started: {self.session_id}")
        return inference_state

    def reset_state(self, inference_state: Dict[str, Any]) -> None:
        """
        Reset the video predictor state.

        WARNING: This method clears ALL objects in the session!
        Use remove_object() instead if you want to remove specific objects.
        Only call this when you want to completely start over.

        Args:
            inference_state: Inference state dictionary to reset
        """
        session_id = inference_state.get("session_id")
        if session_id is None:
            return

        # WARNING: This resets the entire session, removing ALL tracked objects
        print(f"WARNING: Resetting session {session_id} - this removes ALL objects!")
        _ = self.video_predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

    def remove_object(self, inference_state: Dict[str, Any], obj_id: int) -> None:
        """
        Remove a specific object from tracking without affecting other objects.

        Args:
            inference_state: Inference state dictionary
            obj_id: Object ID to remove
        """
        session_id = inference_state.get("session_id")
        if session_id is None:
            return

        _ = self.video_predictor.handle_request(
            request=dict(
                type="remove_object",
                session_id=session_id,
                obj_id=obj_id,
            )
        )

    def save_frames_to_directory(self, frames: np.ndarray, output_dir: str) -> List[str]:
        """
        Save video frames to a temporary directory for SAM3 video predictor.

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

    def add_new_mask_with_text(
        self,
        inference_state: Dict[str, Any],
        frame_idx: int,
        text_prompt: str,
        obj_id: Optional[int] = None
    ) -> Tuple[int, List[int], Dict]:
        """
        Add a new mask using text prompt at a specific frame.

        Args:
            inference_state: Inference state dictionary
            frame_idx: Frame index to add mask
            text_prompt: Text description of object
            obj_id: Optional object ID (if not provided, SAM3 assigns one)

        Returns:
            Tuple of (frame_idx, obj_ids, outputs)
        """
        session_id = inference_state.get("session_id")

        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "text": text_prompt,
        }

        if obj_id is not None:
            request["obj_id"] = obj_id

        response = self.video_predictor.handle_request(request=request)
        outputs = response["outputs"]

        # Extract object IDs from outputs
        obj_ids = [obj_info["id"] for obj_info in outputs]

        return frame_idx, obj_ids, outputs

    def add_new_mask(
        self,
        inference_state: Dict[str, Any],
        frame_idx: int,
        obj_id: int,
        mask: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Add a new mask at a specific frame using SAM3's native mask prompt support.

        This method directly uses SAM3's tracker.add_new_mask() to preserve the
        full mask information, avoiding the information loss from point conversion.

        Args:
            inference_state: Inference state dictionary containing the actual SAM3 state
            frame_idx: Frame index to add mask
            obj_id: Object ID for this mask
            mask: Binary mask tensor (H, W)

        Returns:
            Tuple of (frame_idx, out_obj_ids, out_mask_logits)
        """
        # Get the actual SAM3 inference state (not the session wrapper)
        sam3_state = inference_state.get("state")

        # Ensure mask is on correct device and dtype
        if isinstance(mask, torch.Tensor):
            mask = mask.to(self.device).float()
        else:
            mask = torch.from_numpy(mask).to(self.device).float()

        # Ensure mask is 2D
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        elif mask.dim() == 4:
            mask = mask.squeeze(0).squeeze(0)

        # Check for empty mask
        if mask.sum() == 0:
            print(f"Warning: Empty mask for object {obj_id}")
            img_height, img_width = mask.shape
            return frame_idx, torch.tensor([obj_id]), torch.zeros((1, 1, img_height, img_width))

        # Use SAM3's native mask input via the underlying tracker model
        # This preserves full mask information without converting to points
        _, out_obj_ids, low_res_masks, video_res_masks = self.video_predictor.model.tracker.add_new_mask(
            inference_state=sam3_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask,
        )

        # Return in SAM2-compatible format
        # video_res_masks is already in shape (1, 1, H, W)
        return frame_idx, out_obj_ids, video_res_masks

    def propagate_in_video(
        self,
        inference_state: Dict[str, Any],
        start_frame_idx: int = 0,
        max_frame_num_to_track: int = None,
        reverse: bool = False
    ) -> Generator[Tuple[int, torch.Tensor, torch.Tensor], None, None]:
        """
        Propagate masks through video frames.

        NOTE: For efficiency, SAM3 supports bidirectional propagation in a single pass
        using propagation_direction="both". The 'reverse' parameter is kept for
        SAM2 API compatibility but using "both" is recommended.

        Args:
            inference_state: Inference state dictionary
            start_frame_idx: Starting frame index for propagation
            max_frame_num_to_track: Maximum number of frames to track
            reverse: Whether to propagate in reverse (for SAM2 compatibility)
                    Note: Consider using propagate_all_objects() with direction="both" instead

        Yields:
            Tuple of (frame_idx, out_obj_ids, out_mask_logits) for each frame
        """
        session_id = inference_state.get("session_id")

        # Determine propagation direction based on reverse flag
        # Note: SAM3 supports "both" for bidirectional propagation in one pass
        if reverse:
            propagation_direction = "backward"
        else:
            propagation_direction = "forward"

        # SAM3 propagates using the direction parameter
        for response in self.video_predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
                start_frame_index=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]

            # Convert outputs to SAM2-compatible format
            if len(outputs) == 0:
                continue

            obj_ids = torch.tensor([obj_info["id"] for obj_info in outputs])

            # Extract masks
            masks = []
            for obj_info in outputs:
                if "mask" in obj_info:
                    masks.append(torch.from_numpy(obj_info["mask"]).float())

            if len(masks) > 0:
                mask_logits = torch.stack(masks).unsqueeze(1)  # (N, 1, H, W)
            else:
                continue

            yield frame_idx, obj_ids, mask_logits

    def add_objects_batch(
        self,
        inference_state: Dict[str, Any],
        frame_idx: int,
        objects: List[Dict[str, Any]]
    ) -> None:
        """
        Add multiple objects at once at a specific frame for efficient tracking.

        Args:
            inference_state: Inference state dictionary
            frame_idx: Frame index to add objects
            objects: List of dicts with keys 'obj_id', 'mask', 'class_name'
        """
        print(f"Adding {len(objects)} objects at frame {frame_idx}")
        for obj_info in objects:
            obj_id = obj_info['obj_id']
            mask = obj_info['mask']
            class_name = obj_info.get('class_name', 'unknown')

            # Add each object using add_new_mask
            self.add_new_mask(
                inference_state,
                frame_idx,
                obj_id,
                mask
            )
            print(f"  Added object {obj_id} ({class_name})")

    def propagate_all_objects(
        self,
        inference_state: Dict[str, Any],
        start_frame: int,
        end_frame: int,
        propagation_direction: str = "both"
    ) -> Dict[int, Dict[int, Dict]]:
        """
        Propagate all tracked objects through a frame range.

        This method efficiently propagates all objects simultaneously,
        leveraging SAM3's multi-object tracking capabilities.

        Args:
            inference_state: Inference state dictionary
            start_frame: Start frame index
            end_frame: End frame index
            propagation_direction: "forward", "backward", or "both"

        Returns:
            Dictionary: frame_idx -> {obj_id -> {'mask', 'class_name', 'mask_size'}}
        """
        all_frame_masks = {i: {} for i in range(start_frame, end_frame + 1)}

        session_id = inference_state.get("session_id")

        # Use SAM3's native propagation_in_video with "both" direction
        for response in self.video_predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
                start_frame_index=start_frame,
                max_frame_num_to_track=end_frame - start_frame + 1,
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]

            if not (start_frame <= frame_idx <= end_frame):
                continue

            for obj_info in outputs:
                obj_id = obj_info["id"]
                if "mask" in obj_info:
                    mask = torch.from_numpy(obj_info["mask"]).float()
                    all_frame_masks[frame_idx][obj_id] = {
                        'mask': mask,
                        'class_name': f'obj_{obj_id}',  # Default, should be overridden
                        'mask_size': mask.sum().item()
                    }

        return all_frame_masks

    def close_session(self, inference_state: Dict[str, Any]) -> None:
        """
        Close the current session to free resources.

        This properly cleans up GPU memory and session state.

        Args:
            inference_state: Inference state dictionary
        """
        session_id = inference_state.get("session_id")
        if session_id is None:
            return

        _ = self.video_predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

        # Clear the state reference
        if "state" in inference_state:
            del inference_state["state"]

        self.session_id = None
        self.video_path = None

        # Force garbage collection to free GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        print(f"SAM3 session closed: {session_id}")

    def shutdown(self) -> None:
        """
        Shutdown the predictor to free up multi-GPU resources.
        """
        if hasattr(self.video_predictor, 'shutdown'):
            self.video_predictor.shutdown()
            print("SAM3 video predictor shutdown")

    def detect_and_track(
        self,
        frames: np.ndarray,
        text_prompt: str,
        step: int = 20,
        iou_threshold: float = 0.8
    ) -> Dict[int, MaskDictionaryModel]:
        """
        High-level interface: detect objects and track them across video.

        Args:
            frames: Video frames as numpy array (N, H, W, 3)
            text_prompt: Text prompt for object detection
            step: Keyframe sampling interval
            iou_threshold: IoU threshold for object matching

        Returns:
            Dictionary mapping frame_idx -> MaskDictionaryModel
        """
        raise NotImplementedError(
            "Use the lower-level methods (init_state, add_new_mask_with_text, propagate_in_video) "
            "for more control. See _process_video_mode_with_continuous_id() for usage example."
        )

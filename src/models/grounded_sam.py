import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, List
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundedSAM2:
    def __init__(
        self,
        grounding_model: str = "IDEA-Research/grounding-dino-base",
        sam2_checkpoint: Optional[str] = None,
        sam2_model_config: Optional[str] = None,
        device: Optional[str] = None,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ):
        """Initialize Grounded SAM2 model.
        
        Args:
            grounding_model: HuggingFace model ID for Grounding DINO
            sam2_checkpoint: Path to SAM2 checkpoint file
            sam2_model_config: Path to SAM2 model config file
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
            box_threshold: Threshold for box detection from Grounding DINO
            text_threshold: Threshold for text matching from Grounding DINO
        """
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Build SAM2 model
        if sam2_checkpoint is not None and sam2_model_config is not None:
            # Use checkpoint/config if provided
            self.sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        else:

            # Use SAM2 class for HuggingFace models (reuses existing implementation)
            self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-large",
                hydra_overrides_extra=["++model.compile_image_encoder=True"],
            )
            
        # Build Grounding DINO from HuggingFace
        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model,
        ).to(self.device)

    # def print_gpu_mem(tag: str = ""):
        
    #     if not torch.cuda.is_available():
    #         print(f"[GPU MEM][{tag}] CUDA not available")
    #         return

    #     device = torch.cuda.current_device()
    #     allocated = torch.cuda.memory_allocated(device) / 1024**2
    #     reserved  = torch.cuda.memory_reserved(device) / 1024**2
    #     total     = torch.cuda.get_device_properties(device).total_memory / 1024**2
    #     free_est  = total - reserved

    #     print(
    #         f"[GPU MEM][{tag}] "
    #         f"allocated={allocated:.1f}MB | "
    #         f"reserved={reserved:.1f}MB | "
    #         f"free(est)={free_est:.1f}MB | "
    #         f"total={total:.1f}MB"
    #     )

    def _preprocess_image(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> Tuple[Image.Image, np.ndarray, Tuple[int, int]]:
        """Preprocess input image for Grounding DINO and SAM2.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format or PIL Image
        
        Returns:
            Tuple of (pil_image, image_np, original_shape):
                - pil_image: PIL Image in RGB format
                - image_np: Numpy array representation of the image
                - original_shape: Original image shape (H, W)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Store original shape for empty result case
            original_shape = image.shape[:2]
            if image.dtype != np.uint8:
                image_array = (image * 255).astype(np.uint8)
            else:
                image_array = image
            pil_image = Image.fromarray(image_array)
        else:
            pil_image = image
            original_shape = pil_image.size[::-1]  # (H, W)
        
        # Ensure RGB format
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy for SAM2
        image_np = np.array(pil_image)
        
        return pil_image, image_np, original_shape
        
    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Run Grounded SAM2 inference.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format or PIL Image
            text_prompt: Text prompt for object detection (should be lowercase and end with a dot)
        
        Returns:
            Tuple of (masks, scores, logits, boxes, labels):
                - masks: Binary masks (n, H, W)
                - scores: Confidence scores from SAM2 (n,)
                - logits: Logits from SAM2 (n, H, W)
                - boxes: Bounding boxes from Grounding DINO (n, 4) in xyxy format
                - labels: Class labels from Grounding DINO (n,)
        """
        # Preprocess image
        image, image_np, original_shape = self._preprocess_image(image)
        
        # Set image for SAM2 predictor
        self.sam2_predictor.set_image(image)
        
        # Run Grounding DINO
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # box_threshold=self.box_threshold,
            # text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        # Get boxes from Grounding DINO results
        if len(results) == 0 or len(results[0]["boxes"]) == 0:
            # No detections found
            return (
                np.array([]).reshape(0, *original_shape),
                np.array([]),
                np.array([]).reshape(0, *original_shape),
                np.array([]).reshape(0, 4),
                []
            )
        
        input_boxes = results[0]["boxes"].cpu().numpy()
        labels = results[0]["labels"]
        confidences = results[0]["scores"].cpu().numpy()
        
        # Run SAM2 segmentation with boxes
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Convert masks shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        return masks, scores, logits, input_boxes, labels


    def reset_predictor(self):
        self.sam2_predictor.reset_predictor()
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw


class VisualTraceVisualizer:
    """Visualization utilities for VisualTracePipeline."""
    
    @staticmethod
    def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """Overlay mask on original image."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        overlay = image.copy()
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
        return Image.fromarray(overlay)
    
    @staticmethod
    def draw_keypoints_on_image(image: Image.Image, keypoints: np.ndarray) -> Image.Image:
        """Draw keypoints on image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        # Use cyan (complementary color of red mask)
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            draw.ellipse([x-3, y-3, x+3, y+3], fill=(0, 255, 255))
        return img
    
    @staticmethod
    def save_visualizations(
        image: np.ndarray,
        mask: np.ndarray,
        keypoints: np.ndarray,
        prefix: str = ""
    ):
        """Save all visualization images."""
        Image.fromarray(image).save(f"{prefix}pre_image.png")
        VisualTraceVisualizer.overlay_mask_on_image(image, mask).save(f"{prefix}post_image.png")
        overlay_img = VisualTraceVisualizer.overlay_mask_on_image(image, mask)
        VisualTraceVisualizer.draw_keypoints_on_image(overlay_img, keypoints).save(f"{prefix}keypoint_post_image.png")


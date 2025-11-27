from typing import Any, Dict
import os

import numpy as np
from PIL import Image, ImageDraw

from src.models.molmo import Molmo
from src.models.sam2 import SAM2

from .base import BasePipeline, load_config
from tqdm import tqdm

def mask_to_bbox(mask: np.ndarray) -> Dict[str, int]:
    """Convert binary mask to bounding box coordinates."""
    if not mask.any():
        return {"x": 0, "y": 0, "width": 0, "height": 0}

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return {
        "x": int(x_min),
        "y": int(y_min),
        "width": int(x_max - x_min + 1),
        "height": int(y_max - y_min + 1)
    }


def visualize_results(frame: np.ndarray, points: np.ndarray = None,
                     masks: np.ndarray = None, bboxes: list = None) -> Image.Image:
    """Visualize points, masks, and bounding boxes on frame.

    Args:
        frame: Input frame (H, W, 3) numpy array
        points: Array of points (N, 2) in (x, y) format
        masks: Array of binary masks (N, H, W)
        bboxes: List of bounding box dicts with x, y, width, height

    Returns:
        PIL Image with visualizations overlaid
    """
    # Convert frame to PIL Image
    vis_img = Image.fromarray(frame.astype(np.uint8)).convert("RGBA")

    # Create overlay for masks
    overlay = Image.new("RGBA", vis_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw masks with semi-transparent colors
    if masks is not None and len(masks) > 0:
        colors = [
            (255, 0, 0, 100),    # Red
            (0, 255, 0, 100),    # Green
            (0, 0, 255, 100),    # Blue
            (255, 255, 0, 100),  # Yellow
            (255, 0, 255, 100),  # Magenta
        ]

        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]
            # Create mask overlay
            mask_img = Image.new("RGBA", vis_img.size, (0, 0, 0, 0))
            mask_pixels = mask_img.load()

            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if mask[y, x]:
                        mask_pixels[x, y] = color

            overlay = Image.alpha_composite(overlay, mask_img)

    # Composite overlay onto image
    vis_img = Image.alpha_composite(vis_img, overlay)

    # Convert back to RGB for drawing
    vis_img = vis_img.convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    # Draw bounding boxes
    if bboxes is not None:
        bbox_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]

        for idx, bbox in enumerate(bboxes):
            if bbox["width"] > 0 and bbox["height"] > 0:
                color = bbox_colors[idx % len(bbox_colors)]
                x1 = bbox["x"]
                y1 = bbox["y"]
                x2 = x1 + bbox["width"]
                y2 = y1 + bbox["height"]

                # Draw rectangle with thick border
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Draw points last so they're on top
    if points is not None and len(points) > 0:
        point_radius = 5
        for point in points:
            x, y = int(point[0]), int(point[1])
            # Draw point as circle with outline
            draw.ellipse([x - point_radius, y - point_radius,
                         x + point_radius, y + point_radius],
                        fill=(255, 255, 0), outline=(0, 0, 0), width=2)

    return vis_img


class AffordanceType1Pipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.molmo = Molmo(**config.get("molmo", {}))
        self.sam2 = SAM2(**config.get("sam2", {}))

    def preprocess(self, data_dict: Dict[str, Any]):
        """Prepare input data for processing."""
        return data_dict

    def process(self, data_dict: Dict[str, Any]):
        """Main pipeline: pointing with Molmo, masking with SAM2, bbox conversion."""
        frames = data_dict["frames"]  # (N, H, W, 3)

        # For one frame example (Remove later)
        frames = frames[0][np.newaxis, :]

        description = f"Point to the part that is used to {data_dict['description']}."

        # Convert numpy frames to PIL Images for Molmo
        pil_images = [Image.fromarray(frame) for frame in frames]

        # Use Molmo to extract points for the task description
        points_per_frame = self.molmo(pil_images, query=description)
        print(points_per_frame)

        # Create save directory for visualizations
        save_dir = self.config.get("save_dir", ".")
        vis_dir = os.path.join(save_dir, "visualizations", data_dict["video_name"])
        os.makedirs(vis_dir, exist_ok=True)

        results = []
        for i, (frame, points) in enumerate(zip(frames, points_per_frame)):
            frame_result = {
                "frame_idx": i,
                "points": points.tolist() if points is not None else None,
                "masks": None,
                "bboxes": None,
                "visualization_path": None
            }

            masks = None
            bboxes = None

            if points is not None and len(points) > 0:
                # Use SAM2 to generate masks from points
                masks, scores, logits = self.sam2(frame, points)

                # Convert masks to bounding boxes
                bboxes = []
                for mask in masks:
                    bbox = mask_to_bbox(mask)
                    bboxes.append(bbox)

                frame_result.update({
                    "masks": masks.tolist(),
                    "scores": scores.tolist(),
                    "bboxes": bboxes
                })

            # Visualize and save
            vis_img = visualize_results(frame, points, masks, bboxes)
            vis_path = os.path.join(vis_dir, f"frame_{i:04d}.png")
            vis_img.save(vis_path)
            frame_result["visualization_path"] = vis_path

            results.append(frame_result)

        return {
            "video_name": data_dict["video_name"],
            "description": description,
            "metadata": data_dict.get("metadata", {}),
            "results": results,
            "visualization_dir": vis_dir
        }


if __name__ == "__main__":
    from src.datasets.egodex import EgoDexDataset
    
    # Load pipeline configuration and create pipeline
    config = load_config("affordance_type1")
    pipeline = AffordanceType1Pipeline(config)
    
    # Load EgoDex dataset and get first sample
    dataset = EgoDexDataset()
    if len(dataset) == 0:
        print("No EgoDex data found. Please check data directory.")
        exit(1)
    
    # Get first video sample
    data_dict = dataset[0]
    print(f"Testing pipeline with video: {data_dict['video_name']}")
    print(f"Description: {data_dict['description']}")
    print(f"Frames shape: {data_dict['frames'].shape}")
    
    # Run pipeline
    print("\nRunning affordance type1 pipeline...")
    results = pipeline(data_dict, save_dir=".")
    
    # Display results
    print(f"\nResults for video: {results['video_name']}")
    print(f"Number of frames processed: {len(results['results'])}")
    
    for frame_result in results['results'][:3]:  # Show first 3 frames
        frame_idx = frame_result['frame_idx']
        points = frame_result['points']
        bboxes = frame_result['bboxes']
        
        print(f"\nFrame {frame_idx}:")
        if points:
            print(f"  Points detected: {len(points)}")
            print(f"  Points: {points}")
        else:
            print("  No points detected")
            
        if bboxes:
            print(f"  Bounding boxes: {bboxes}")
        else:
            print("  No bounding boxes generated")

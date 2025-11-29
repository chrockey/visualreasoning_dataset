from typing import Any, Dict, List, Tuple
import os

import numpy as np
from PIL import Image, ImageDraw
import cv2


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

def compute_overlap(bbox: np.ndarray, mask: np.ndarray) -> float:
    """Compute overlap area between bounding box and mask.

    Args:
        bbox: Bounding box in xyxy format [x1, y1, x2, y2]
        mask: Binary mask (H, W)

    Returns:
        Overlap area (number of pixels)
    """
    x1, y1, x2, y2 = bbox.astype(int)
    # Crop mask to bbox region
    bbox_mask_region = mask[y1:y2, x1:x2]
    # Count overlapping pixels
    overlap_area = np.sum(bbox_mask_region)
    return float(overlap_area)


def find_best_interacting_object(
    masks: np.ndarray,
    boxes: np.ndarray,
    labels: List[str]
) -> Tuple[int, List[int], List[int], float]:
    """Find the object with maximum overlap with hand/gripper.

    Args:
        masks: All detected masks (N, H, W)
        boxes: All detected bounding boxes (N, 4) in xyxy format
        labels: List of labels for each detection

    Returns:
        Tuple of (best_object_idx, object_indices, hand_gripper_indices, max_overlap):
            - best_object_idx: Index of object with maximum overlap with hand/gripper (or None)
            - object_indices: List of indices for all detected objects
            - hand_gripper_indices: List of indices for all detected hands/grippers
            - max_overlap: Maximum overlap area in pixels
    """
    # Find object and hand/gripper indices
    object_indices = []
    hand_gripper_indices = []

    for idx, label in enumerate(labels):
        label_lower = label.lower()
        is_hand_or_gripper = any(keyword in label_lower for keyword in ['hand', 'gripper', 'finger'])

        if is_hand_or_gripper:
            hand_gripper_indices.append(idx)
        else:
            object_indices.append(idx)

    # Check for overlap
    best_object_idx = None
    max_overlap = 0.0

    if len(object_indices) > 0 and len(hand_gripper_indices) > 0:
        # Combine hand/gripper masks
        combined_hg_mask = np.zeros_like(masks[0], dtype=bool)
        for hg_idx in hand_gripper_indices:
            combined_hg_mask = np.logical_or(combined_hg_mask, masks[hg_idx])

        # Find object with maximum overlap
        for obj_idx in object_indices:
            overlap = compute_overlap(boxes[obj_idx], combined_hg_mask)
            if overlap > max_overlap:
                max_overlap = overlap
                best_object_idx = obj_idx

    return best_object_idx, object_indices, hand_gripper_indices, max_overlap


def sample_interaction_points(
    object_mask: np.ndarray,
    hand_gripper_masks: List[np.ndarray],
    num_points: int = 5
) -> np.ndarray:
    """Sample interaction points from object mask where it overlaps with hand/gripper.

    Args:
        object_mask: Binary mask of the object (H, W)
        hand_gripper_masks: List of hand/gripper masks (H, W)
        num_points: Number of points to sample

    Returns:
        Array of sampled points (N, 2) in (x, y) format, where N <= num_points
    """
    # Combine all hand/gripper masks
    combined_hand_mask = np.zeros_like(object_mask, dtype=bool)
    for hg_mask in hand_gripper_masks:
        combined_hand_mask = np.logical_or(combined_hand_mask, hg_mask)

    # Find overlap region
    overlap_region = np.logical_and(object_mask, combined_hand_mask)

    # Get coordinates of overlap pixels
    overlap_coords = np.argwhere(overlap_region)  # Returns (y, x) format

    if len(overlap_coords) == 0:
        # No overlap, sample from object mask instead
        object_coords = np.argwhere(object_mask)
        if len(object_coords) == 0:
            return np.array([])
        overlap_coords = object_coords

    # Sample points
    num_points = min(num_points, len(overlap_coords))
    sampled_indices = np.random.choice(len(overlap_coords), size=num_points, replace=False)
    sampled_coords = overlap_coords[sampled_indices]

    # Convert from (y, x) to (x, y) format
    sampled_points = sampled_coords[:, [1, 0]]

    return sampled_points


def create_demo_video(
    vis_dir: str,
    output_path: str,
    fps: int = 10,
    first_interaction_frame: int = None,
    caption: str = None
) -> str:
    """Create demo video from visualization frames.

    Args:
        vis_dir: Directory containing frame_XXXX.png files
        output_path: Path to save output video
        fps: Frames per second
        first_interaction_frame: Frame index where interaction starts (for annotation)
        caption: Optional text caption to overlay on video

    Returns:
        Path to created video
    """
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(vis_dir) if f.startswith('frame_') and f.endswith('.png')])

    if len(frame_files) == 0:
        raise ValueError(f"No frame files found in {vis_dir}")

    # Read first frame to get dimensions
    first_frame_path = os.path.join(vis_dir, frame_files[0])
    first_img = cv2.imread(first_frame_path)
    height, width, _ = first_img.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating demo video with {len(frame_files)} frames...")

    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(vis_dir, frame_file)
        frame = cv2.imread(frame_path)

        # Add overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6  # Smaller font to avoid occlusion
        font_thickness = 2

        # Caption at the top (if provided)
        if caption:
            # Use smaller font for caption
            caption_font_scale = 0.5
            caption_thickness = 1
            caption_size = cv2.getTextSize(caption, font, caption_font_scale, caption_thickness)[0]
            caption_x = 10
            caption_y = 20

            # Draw semi-transparent background for caption
            overlay = frame.copy()
            cv2.rectangle(overlay, (caption_x - 5, caption_y - caption_size[1] - 5),
                         (caption_x + caption_size[0] + 5, caption_y + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, caption, (caption_x, caption_y), font, caption_font_scale,
                       (255, 255, 255), caption_thickness)

        # Frame number at bottom-left
        text = f"Frame {idx}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = 10
        text_y = height - 10

        # Draw background rectangle for text
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        # Mark interaction frame at bottom
        if first_interaction_frame is not None and idx == first_interaction_frame:
            text2 = "INTERACTION"
            text_size2 = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]
            text2_x = text_x + text_size[0] + 20
            text2_y = height - 10

            cv2.rectangle(frame, (text2_x - 5, text2_y - text_size2[1] - 5),
                         (text2_x + text_size2[0] + 5, text2_y + 5), (0, 255, 255), -1)
            cv2.putText(frame, text2, (text2_x, text2_y), font, font_scale, (0, 0, 0), font_thickness)

        video_writer.write(frame)

    video_writer.release()
    print(f"Demo video saved to: {output_path}")

    return output_path


def visualize_affordance(
    frame: np.ndarray,
    bboxes: List[np.ndarray],
    masks: List[np.ndarray],
    labels: List[str] = None,
    interaction_points: np.ndarray = None
) -> Image.Image:
    """Visualize affordance detection results.

    - Main object: bounding box only
    - Hand/Gripper: semi-transparent mask overlay
    - Interaction points: yellow circles

    Args:
        frame: Input frame (H, W, 3) numpy array
        bboxes: List of bounding boxes in xyxy format
        masks: List of binary masks (H, W)
        labels: List of labels for each detection
        interaction_points: Array of interaction points (N, 2) in (x, y) format

    Returns:
        PIL Image with visualizations overlaid
    """
    # Convert frame to PIL Image
    vis_img = Image.fromarray(frame.astype(np.uint8)).convert("RGBA")

    # Create overlay for masks
    overlay = Image.new("RGBA", vis_img.size, (0, 0, 0, 0))

    # Define colors
    object_color = (255, 0, 0)      # Red for main object bbox
    hand_color = (0, 255, 0, 120)   # Green semi-transparent for hand
    gripper_color = (0, 0, 255, 120) # Blue semi-transparent for gripper
    point_color = (255, 255, 0)     # Yellow for interaction points

    # Process each detection
    for idx, (bbox, mask, label) in enumerate(zip(bboxes, masks, labels)):
        label_lower = label.lower()

        # Check if this is hand/gripper or main object
        is_hand_or_gripper = any(keyword in label_lower for keyword in ['hand', 'gripper', 'finger'])

        if is_hand_or_gripper:
            # Draw mask for hand/gripper
            color = hand_color if 'hand' in label_lower else gripper_color
            mask_img = Image.new("RGBA", vis_img.size, (0, 0, 0, 0))
            mask_pixels = mask_img.load()

            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if mask[y, x]:
                        mask_pixels[x, y] = color

            overlay = Image.alpha_composite(overlay, mask_img)

    # Composite mask overlay onto image
    vis_img = Image.alpha_composite(vis_img, overlay)
    vis_img = vis_img.convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    # Draw bounding boxes for main objects
    for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
        label_lower = label.lower()
        is_hand_or_gripper = any(keyword in label_lower for keyword in ['hand', 'gripper', 'finger'])

        if not is_hand_or_gripper:
            # Draw bounding box for main object
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=object_color, width=3)

            # Draw label
            text_bbox = draw.textbbox((x1, y1 - 20), label)
            draw.rectangle(text_bbox, fill=object_color)
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255))

    # Draw interaction points
    if interaction_points is not None and len(interaction_points) > 0:
        point_radius = 6
        for point in interaction_points:
            x, y = int(point[0]), int(point[1])
            # Draw circle with outline
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=point_color,
                outline=(0, 0, 0),
                width=2
            )

    return vis_img


def visualize_video_frame(
    frame: np.ndarray,
    mask: np.ndarray = None,
    bbox: Dict[str, int] = None,
    interaction_points: np.ndarray = None
) -> Image.Image:
    """Visualize a single video frame with mask, bbox, and interaction points.

    Args:
        frame: Input frame (H, W, 3) numpy array
        mask: Binary mask (H, W) - optional
        bbox: Bounding box dict with keys 'x', 'y', 'width', 'height' - optional
        interaction_points: Array of interaction points (N, 2) in (x, y) format - optional

    Returns:
        PIL Image with visualizations overlaid
    """
    vis_img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    # Draw mask if provided
    if mask is not None:
        # Create semi-transparent overlay
        overlay = Image.new("RGBA", vis_img.size, (0, 0, 0, 0))

        mask_img = Image.new("RGBA", vis_img.size, (0, 0, 0, 0))
        mask_pixels = mask_img.load()
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    mask_pixels[x, y] = (255, 0, 0, 120)  # Red semi-transparent

        vis_img = vis_img.convert("RGBA")
        vis_img = Image.alpha_composite(vis_img, mask_img)
        vis_img = vis_img.convert("RGB")
        draw = ImageDraw.Draw(vis_img)

    # Draw bounding box if provided
    if bbox is not None and bbox.get('width', 0) > 0 and bbox.get('height', 0) > 0:
        x1 = bbox['x']
        y1 = bbox['y']
        x2 = x1 + bbox['width']
        y2 = y1 + bbox['height']
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

    # Draw interaction points if provided
    if interaction_points is not None:
        point_radius = 6
        for point in interaction_points:
            x, y = int(point[0]), int(point[1])
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=(255, 255, 0),
                outline=(0, 0, 0),
                width=2
            )

    return vis_img
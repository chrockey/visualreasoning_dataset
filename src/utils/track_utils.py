import numpy as np
import torch
from typing import Dict, List


def suppress_overlapping_masks(
    all_frame_masks: Dict[int, Dict[int, Dict]],
    num_frames: int,
    overlap_threshold: float = 0.3
) -> Dict[int, Dict[int, Dict]]:
    """
    Suppress overlapping masks, favoring smaller masks.

    When two masks overlap significantly, keep the smaller one.
    This prevents objects from being over-segmented.

    Args:
        all_frame_masks: Dictionary of frame_idx -> {obj_id -> mask_info}
        num_frames: Total number of frames
        overlap_threshold: IoU threshold for considering masks as overlapping

    Returns:
        Filtered dictionary with suppressed overlaps
    """
    suppressed_masks = {}

    for frame_idx in range(num_frames):
        frame_masks = all_frame_masks.get(frame_idx, {})

        if len(frame_masks) <= 1:
            # No overlaps possible
            suppressed_masks[frame_idx] = frame_masks
            continue

        # Sort objects by mask size (smallest first)
        sorted_objects = sorted(
            frame_masks.items(),
            key=lambda x: x[1]['mask_size']
        )

        # Keep track of which objects to keep
        keep_objects = {}
        suppressed_count = 0

        for obj_id, obj_info in sorted_objects:
            # Check overlap with already kept objects
            should_keep = True

            for kept_id, kept_info in keep_objects.items():
                # Calculate IoU
                mask1 = obj_info['mask'].to(torch.float32)
                mask2 = kept_info['mask'].to(torch.float32)

                intersection = (mask1 * mask2).sum()
                union = mask1.sum() + mask2.sum() - intersection

                if union > 0:
                    iou = intersection / union

                    if iou > overlap_threshold:
                        # Significant overlap - suppress the larger mask
                        if obj_info['mask_size'] > kept_info['mask_size']:
                            should_keep = False
                            suppressed_count += 1
                            break
                        else:
                            # Current object is smaller, remove the kept one
                            del keep_objects[kept_id]
                            suppressed_count += 1

            if should_keep:
                keep_objects[obj_id] = obj_info

        suppressed_masks[frame_idx] = keep_objects

        if suppressed_count > 0:
            print(f"  Frame {frame_idx}: suppressed {suppressed_count} overlapping masks")

    return suppressed_masks


def sample_points_from_masks(masks, num_points):
    """
    sample points from masks and return its absolute coordinates

    Args:
        masks: np.array with shape (n, h, w)
        num_points: int

    Returns:
        points: np.array with shape (n, points, 2)
    """
    n, h, w = masks.shape
    points = []

    for i in range(n):
        # find the valid mask points
        indices = np.argwhere(masks[i] == 1)  
        # the output format of np.argwhere is (y, x) and the shape is (num_points, 2)
        # we should convert it to (x, y)
        indices = indices[:, ::-1]  # (num_points, [y x]) to (num_points, [x y])
        
        # import pdb; pdb.set_trace()
        if len(indices) == 0:
            # if there are no valid points, append an empty array
            points.append(np.array([]))
            continue
        
        # resampling if there's not enough points
        if len(indices) < num_points:
            sampled_indices = np.random.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = np.random.choice(len(indices), num_points, replace=False)
        
        sampled_points = indices[sampled_indices]
        points.append(sampled_points)

    # convert to np.array
    points = np.array(points, dtype=np.float32)
    return points

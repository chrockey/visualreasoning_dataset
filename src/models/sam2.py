from typing import Literal

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor


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

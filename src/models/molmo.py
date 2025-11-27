import re
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def extract_points(
    molmo_output: str, image_wh: Tuple[int, int], return_all: bool = False
) -> np.array:
    """
    Obtained from https://huggingface.co/allenai/Molmo-7B-O-0924/discussions/1
    """
    image_w, image_h = image_wh
    all_points = []
    for match in re.finditer(
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
        molmo_output,
    ):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            points = all_points.append(point)

    if len(all_points) > 0:
        points = np.stack(all_points, axis=0)
        if not return_all:
            points = points[:1]  # pick the first point
    else:
        points = None
    return points


class Molmo:
    def __init__(
        self,
        molmo_model_id: str = "allenai/Molmo-7B-D-0924",
        use_tqdm: bool = False,
        max_new_tokens: int = 64,
        seed: int = 42,
    ):
        self.molmo_model_id = molmo_model_id
        self.use_tqdm = use_tqdm
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.configure()

    def configure(self):
        # Load the processor and model
        model = AutoModelForCausalLM.from_pretrained(
            self.molmo_model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        model = model.to(dtype=torch.bfloat16)
        compiled_model = torch.compile(model)
        self.molmo = {
            "processor": AutoProcessor.from_pretrained(
                self.molmo_model_id,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            ),
            "model": compiled_model,
        }

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def inference(self, image, text):
        if self.molmo is None:
            raise ValueError("Molmo model is not initialized")

        inputs = self.molmo["processor"].process(images=[image], text=text)
        inputs = {
            k: v.to(self.molmo["model"].device).unsqueeze(0) for k, v in inputs.items()
        }
        output = self.molmo["model"].generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                stop_strings="<|endoftext|>",
            ),
            tokenizer=self.molmo["processor"].tokenizer,
        )
        generated_text = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.molmo["processor"].tokenizer.decode(
            generated_text, skip_special_tokens=True
        )
        image_wh = image.size
        points = extract_points(generated_text, image_wh)
        return points

    def __call__(
        self,
        images: List[Image.Image],
        query: str = "Point to the part of the chair that human sit on.",
    ):
        points_all = []
        for image in tqdm(images, desc="Processing images", disable=not self.use_tqdm):
            points = self.inference(image, query)
            points_all.append(points)

        return points_all


if __name__ == "__main__":
    molmo = Molmo()
    points = molmo(
        image_paths=[f"demo/16341/campos_512_v4/{i:05d}/{i:05d}.png" for i in range(5)],
        query="Point to the part of the chair that human sit on.",
    )
    import pdb

    pdb.set_trace()

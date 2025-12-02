import kagglehub
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

def build_object_extraction_prompt(instruction: str) -> str:
    return f"""
    You are a tool that extracts the main manipulated objects from robot manipulation instructions.

    - The "main manipulated objects" is the physical objects the robot is supposed to grasp, move, push, pull, pick up, or place.
    - Ignore locations, surfaces, containers, and reference objects (tables, shelves, drawers, boxes, rooms, positions).
    - Ignore adjectives that are not needed to identify the object category (e.g., "red cup" -> "cup", "blue mug" -> "mug").
    - Fix minor typos if needed.
    - Output only the noun phrase for the main manipulated object.
    - Use lowercase, no articles, no punctuation, no extra words.
    - If multiple main manipulated objects exists, separate with dot.

    Examples:
    Instruction: "Move the moka pot to the right side of the drawer."
    Answer: moka pot

    Instruction: "Pick up the red mug on the table."
    Answer: mug

    Instruction: "Place the cereal box inside the cupboard."
    Answer: cereal box

    Instruction: "Push the chair closer to the desk."
    Answer: chair

    Instruction: "Open the fridge door."
    Answer: fridge door

    Instruction: "Move the towel from the bed to the chair."
    Answer: towel

    Instruction: "Add lids onto four cups placed on a wooden table with a red background."
    Answer: lid. cup

    Now extract the main manipulated objects.

    Instruction: "{instruction}"
    Answer:
    """

class Gemma:
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        
    def __call__(self, prompt: str):
        prompt = build_object_extraction_prompt(prompt)
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
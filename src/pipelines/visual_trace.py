from typing import Any, Dict
from .base import BasePipeline, load_config
from ..models.gemma import Gemma
from ..models.grounded_sam import GroundedSAM2
from ..models.cotracker import CoTracker
from ..visualizers.visual_trace import VisualTraceVisualizer

# TODO : Implement the VisualTracePipeline
# 1. Load the Gemma, CoTracker v3, Grounded-SAM2 models
# 2. extract main object from the image using Gemma
# 3. Grounded SAM2 to segment the main object and extract the keypoints
# 4. track the keypoints using CoTracker v3
# 5. filter the keypoints using the confidence score

class VisualTracePipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        super().__init__(config)
        self.object_extractor = Gemma(config["gemma"]["model_id"])
        
        # Initialize GroundedSam with config
        grounded_sam_config = config.get("grounded_sam", {})
        self.grounded_segmenter = GroundedSAM2(**grounded_sam_config)
        
        self.keypoint_tracker = CoTracker(config["cotracker"]["model_id"])
        self.verbose = verbose
        self.visualizer = VisualTraceVisualizer()
        
    def preprocess(self, data_dict: Dict[str, Any]):
        raise NotImplementedError

    def process(self, data_dict: Dict[str, Any]):
        descriptions = data_dict["description"]
        frame_sets = data_dict["frames"]
        
        for description, frame_set in zip(descriptions, frame_sets):
            word = self.object_extractor(description) + "."
            image = frame_set[0]
            
            masks, scores, logits, boxes, labels = self.grounded_segmenter(image, word)
            self.grounded_segmenter.reset_predictor()
            
            # if no masks found, skip the frame           
            if masks.shape[0] == 0: continue
            
            # Extract keypoints from masks
            keypoints = CoTracker.extract_keypoints_from_masks(masks)  # (n, 3, 2)
            # TODO: Pass keypoints to keypoint_tracker
            tracked_keypoints, tracked_visibility = self.keypoint_tracker(frame_set, keypoints.reshape(1, -1, 2))
            
            if self.verbose:
                print(description)
                print(word)
                print(masks.shape, scores.shape, logits.shape, boxes.shape)
                print(f"Keypoints shape: {keypoints.shape}")
                print(f"Keypoints for first mask: {keypoints[0]}")
                
                self.visualizer.save_visualizations(image, masks[0], keypoints[0])
                print(tracked_keypoints.shape, tracked_visibility.shape)
            
if __name__ == "__main__":
    
    from src.datasets.agibotworld import AgiBotWorldDataset
    ds = AgiBotWorldDataset("dataset/AgiBotWorld-Beta")
    
    # get first video
    data_dict = ds[0]

    # load config
    config = load_config("visual_trace")
    pipeline = VisualTracePipeline(config, verbose=True)

    # run pipeline
    results = pipeline.process(data_dict)

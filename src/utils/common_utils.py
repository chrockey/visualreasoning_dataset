import os
import json
import cv2
import numpy as np
from dataclasses import dataclass
import supervision as sv
import random

class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")

    @staticmethod
    def draw_masks_and_box_with_supervision(raw_image_path, mask_path, json_path, output_path, verbose=False):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            
            # get each mask from unique mask file
            all_object_masks = []
            for uid in unique_ids:
                if uid == 0: # skip background id
                    continue
                else:
                    object_mask = (mask == uid)
                    all_object_masks.append(object_mask[None])
            
            if len(all_object_masks) == 0:
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
            # get n masks: (n, h, w)
            all_object_masks = np.concatenate(all_object_masks, axis=0)
            
            # load box information
            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            
            all_object_boxes = []
            all_object_ids = []
            all_class_names = []
            object_id_to_name = {}
            with open(file_path, "r") as file:
                json_data = json.load(file)
                for obj_id, obj_item in json_data["labels"].items():
                    # box id
                    instance_id = obj_item["instance_id"]
                    if instance_id not in unique_ids: # not a valid box
                        continue
                    # box coordinates
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    all_object_boxes.append([x1, y1, x2, y2])
                    # box name
                    class_name = obj_item["class_name"]
                    
                    # build id list and id2name mapping
                    all_object_ids.append(instance_id)
                    all_class_names.append(class_name)
                    object_id_to_name[instance_id] = class_name
            
            # Adjust object id and boxes to ascending order
            paired_id_and_box = zip(all_object_ids, all_object_boxes, all_class_names)
            sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])
            
            # Because we get the mask data as ascending order, so we also need to ascend box and ids
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_object_boxes = [pair[1] for pair in sorted_pair]
            all_class_names = [pair[2] for pair in sorted_pair]
            
            detections = sv.Detections(
                xyxy=np.array(all_object_boxes),
                mask=all_object_masks,
                class_id=np.array(all_object_ids, dtype=np.int32),
            )
            
            # custom label to show both id and class name
            labels = [
                f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_class_names)
            ]
            
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, annotated_frame)

    @staticmethod
    def draw_masks_and_box(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            colors = {uid: CommonUtils.random_color() for uid in unique_ids}
            colors[0] = (0, 0, 0)  # background color

            # apply mask to image in RBG channels
            colored_mask = np.zeros_like(image)
            for uid in unique_ids:
                colored_mask[mask == uid] = colors[uid]
            alpha = 0.5  # 调整 alpha 值以改变透明度
            output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                # Draw bounding boxes and labels
                for obj_id, obj_item in json_data["labels"].items():
                    # Extract data from JSON
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    instance_id = obj_item["instance_id"]
                    class_name = obj_item["class_name"]

                    # Draw rectangle
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put text
                    label = f"{instance_id}: {class_name}"
                    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Save the modified image
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, output_image)

    @staticmethod
    def random_color():
        """random color generator"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    @staticmethod
    def draw_visual_trace(frame_dir, json_data_dir, result_dir):
        """
        Draw visual traces of object centroids across frames.
        For each frame, draws lines connecting centroids from all previous frames.

        Args:
            frame_dir: Directory containing raw frame images
            json_data_dir: Directory containing JSON files with centroid information
            result_dir: Directory to save output images with traces
        """
        CommonUtils.creat_dirs(result_dir)

        # Get sorted list of frames
        frame_names = os.listdir(frame_dir)
        frame_names.sort()

        # Build centroid history for each object across all frames
        # Structure: {object_id: [(frame_idx, centroid_x, centroid_y), ...]}
        object_centroid_history = {}

        for frame_idx, frame_name in enumerate(frame_names):
            json_file = os.path.join(json_data_dir, "mask_" + frame_name.split(".")[0] + ".json")

            if not os.path.exists(json_file):
                continue

            with open(json_file, 'r') as f:
                json_data = json.load(f)

                for obj_id_str, obj_item in json_data.get("labels", {}).items():
                    obj_id = int(obj_id_str)
                    centroid_x = obj_item.get("centroid_x", None)
                    centroid_y = obj_item.get("centroid_y", None)

                    # Skip if centroid data is missing
                    if centroid_x is None or centroid_y is None:
                        continue

                    # Initialize history for this object if needed
                    if obj_id not in object_centroid_history:
                        object_centroid_history[obj_id] = []

                    # Add centroid to history
                    object_centroid_history[obj_id].append((frame_idx, centroid_x, centroid_y))

        # Generate unique colors for each object
        object_colors = {obj_id: CommonUtils.random_color() for obj_id in object_centroid_history.keys()}
        for frame_idx, frame_name in enumerate(frame_names):
            # Load original frame
            frame_path = os.path.join(frame_dir, frame_name)
            image = cv2.imread(frame_path)

            if image is None:
                print(f"Warning: Could not load frame {frame_path}")
                continue

            # Draw traces for each object up to current frame
            for obj_id, centroid_list in object_centroid_history.items():
                # Filter centroids up to current frame
                past_centroids = [(cx, cy) for (fidx, cx, cy) in centroid_list if fidx <= frame_idx]

                if len(past_centroids) < 2:
                    # Need at least 2 points to draw a line
                    # Draw a circle at single point
                    if len(past_centroids) == 1:
                        cx, cy = past_centroids[0]
                        cv2.circle(image, (int(cx), int(cy)), 3, object_colors[obj_id], -1)
                    continue

                # Draw lines connecting consecutive centroids
                color = object_colors[obj_id]
                for i in range(len(past_centroids) - 1):
                    pt1 = (int(past_centroids[i][0]), int(past_centroids[i][1]))
                    pt2 = (int(past_centroids[i+1][0]), int(past_centroids[i+1][1]))
                    cv2.line(image, pt1, pt2, color, 2)

                # Draw circles at each centroid point
                for cx, cy in past_centroids:
                    cv2.circle(image, (int(cx), int(cy)), 3, color, -1)

            # Save annotated frame
            output_path = os.path.join(result_dir, frame_name)
            cv2.imwrite(output_path, image)

        print(f"Visual traces saved to {result_dir}")

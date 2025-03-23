import os
import json
import shutil
from pycocotools.coco import COCO
import numpy as np

# Paths
RAW_DATA_DIR = "data/raw/coco"
PROCESSED_DIR = "data/processed/coco_occluded_vehicles"
ANNOTATION_FILE = f"{RAW_DATA_DIR}/annotations/instances_test2017.json"
IMAGE_DIR = f"{RAW_DATA_DIR}/test2017"

# Vehicle category IDs in MS-COCO: car (3), bus (6), truck (8)
VEHICLE_CAT_IDS = [3, 6, 8]

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x_min, y_min, width, height]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def filter_occluded_vehicles():
    """Filter MS-COCO for occluded vehicle images."""
    # Create output directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("data/annotations", exist_ok=True)

    # Load COCO dataset
    coco = COCO(ANNOTATION_FILE)
    img_ids = coco.getImgIds(catIds=VEHICLE_CAT_IDS)
    print(f"Found {len(img_ids)} images with vehicles.")

    # Store filtered data
    filtered_data = {"images": [], "annotations": [], "categories": coco.loadCats(VEHICLE_CAT_IDS)}
    annotation_id = 1

    for img_id in img_ids[:1000]:  # Limit to 1000 images for now
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=VEHICLE_CAT_IDS)
        anns = coco.loadAnns(ann_ids)

        # Check for occlusion by IoU with other objects
        all_anns = coco.getAnnIds(imgIds=img_id)
        all_objects = coco.loadAnns(all_anns)
        vehicle_boxes = [ann["bbox"] for ann in anns if ann["category_id"] in VEHICLE_CAT_IDS]
        other_boxes = [ann["bbox"] for ann in all_objects if ann["category_id"] not in VEHICLE_CAT_IDS]

        is_occluded = False
        for v_box in vehicle_boxes:
            for o_box in other_boxes:
                iou = calculate_iou(v_box, o_box)
                if iou > 0.3:  # Threshold for occlusion
                    is_occluded = True
                    break
            if is_occluded:
                break

        if is_occluded:
            # Copy image
            src_path = f"{IMAGE_DIR}/{img_info['file_name']}"
            dst_path = f"{PROCESSED_DIR}/{img_info['file_name']}"
            shutil.copy(src_path, dst_path)

            # Add to filtered data
            filtered_data["images"].append(img_info)
            for ann in anns:
                if ann["category_id"] in VEHICLE_CAT_IDS:
                    ann["id"] = annotation_id
                    annotation_id += 1
                    filtered_data["annotations"].append(ann)

    # Save annotations
    with open("data/annotations/coco_occluded_vehicles.json", "w") as f:
        json.dump(filtered_data, f)
    print(f"Filtered {len(filtered_data['images'])} occluded vehicle images.")

if __name__ == "__main__":
    filter_occluded_vehicles()
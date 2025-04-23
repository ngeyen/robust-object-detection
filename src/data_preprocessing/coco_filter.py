import os
import json
import shutil
from pycocotools.coco import COCO
import numpy as np

# Paths
RAW_DATA_DIR = "data/raw/coco"
PROCESSED_DIR = "data/processed/coco_occluded_kitchen"
ANNOTATION_FILE = f"{RAW_DATA_DIR}/annotations/instances_train2017.json"
IMAGE_DIR = f"{RAW_DATA_DIR}/train2017"
OUTPUT_ANNOTATION_FILE = "data/annotations/coco_occluded_kitchen.json"

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

def filter_occluded_kitchen():
    """Filter MS-COCO for occluded kitchen images using the 'kitchen' supercategory."""
    # Create output directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("data/annotations", exist_ok=True)

    # Load COCO dataset
    coco = COCO(ANNOTATION_FILE)

    # Debug: Basic stats
    print(f"Total images: {len(coco.imgs)}")
    print(f"Total annotations: {len(coco.anns)}")

    # Get all categories and filter by 'kitchen' supercategory
    all_cats = coco.loadCats(coco.getCatIds())
    kitchen_cat_ids = [cat["id"] for cat in all_cats if cat["supercategory"] == "kitchen"]
    print(f"Kitchen category IDs: {kitchen_cat_ids}")

    # Get kitchen annotations and unique image IDs
    kitchen_anns = [ann for ann in coco.anns.values() if ann["category_id"] in kitchen_cat_ids]

    kitchen_img_ids = set(ann["image_id"] for ann in kitchen_anns)
    print(f"Found {len(kitchen_img_ids)} unique images with kitchen annotations.")

    if not kitchen_img_ids:
        print("Error: No kitchen images found in annotations.")
        return

    # Store filtered data
    filtered_data = {"images": [], "annotations": [], "categories": coco.loadCats(kitchen_cat_ids)}
    annotation_id = 1

    # Process up to 1000 images
    for img_id in list(kitchen_img_ids)[:1000]:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=kitchen_cat_ids)
        anns = coco.loadAnns(ann_ids)

        # Check for occlusion by IoU with other objects
        all_anns = coco.getAnnIds(imgIds=img_id)
        all_objects = coco.loadAnns(all_anns)
        kitchen_boxes = [ann["bbox"] for ann in anns if ann["category_id"] in kitchen_cat_ids]
        other_boxes = [ann["bbox"] for ann in all_objects if ann["category_id"] not in kitchen_cat_ids]

        is_occluded = False
        for k_box in kitchen_boxes:
            for o_box in other_boxes:
                iou = calculate_iou(k_box, o_box)
                if iou > 0.3:  # Threshold for occlusion
                    is_occluded = True
                    break
            if is_occluded:
                break

        if is_occluded:
            # Copy image
            src_path = f"{IMAGE_DIR}/{img_info['file_name']}"
            dst_path = f"{PROCESSED_DIR}/{img_info['file_name']}"
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Image {src_path} not found.")
                continue

            # Add to filtered data
            filtered_data["images"].append(img_info)
            for ann in anns:
                if ann["category_id"] in kitchen_cat_ids:
                    ann["id"] = annotation_id
                    annotation_id += 1
                    filtered_data["annotations"].append(ann)

    # Save annotations
    with open(OUTPUT_ANNOTATION_FILE, "w") as f:
        json.dump(filtered_data, f)
    print(f"Filtered {len(filtered_data['images'])} occluded kitchen images.")

if __name__ == "__main__":
    filter_occluded_kitchen()
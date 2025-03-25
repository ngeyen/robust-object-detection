import os
import torch
import json
from src.models import yolov5_nano
from src.evaluation.compute_metrics import evaluate_predictions, format_predictions_yolov5

PROCESSED_DIR = "data/processed/coco_occluded_vehicles"
GT_FILE = "data/annotations/coco_occluded_vehicles.json"
OUTPUT_BASE = "experiments/baseline_tests/yolov5_nano"

def run_inference():
    with open(GT_FILE, 'r') as f:
        gt_data = json.load(f)
    gt_img_info = {img['id']: img for img in gt_data['images']}
    gt_img_ids = set(gt_img_info.keys())
    gt_annotations = {img_id: [] for img_id in gt_img_ids}
    for ann in gt_data['annotations']:
        gt_annotations[ann['image_id']].append({
            'bbox': [ann['bbox'][0], ann['bbox'][1], 
                     ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]],  # [x_min, y_min, x_max, y_max]
            'category_id': ann['category_id']
        })
    
    print(f"Ground truth image IDs (sample): {list(gt_img_ids)[:5]}")
    print(f"Total ground truth annotations: {len(gt_data['annotations'])}")
    
    pred_dict = {}
    
    for img_file in os.listdir(PROCESSED_DIR):
        img_path = os.path.join(PROCESSED_DIR, img_file)
        img_id = int(img_file.split('.')[0].lstrip('0') or '0')
        print(f"\nProcessing {img_file} -> ID: {img_id}")
        print(f"Image {img_id} dimensions: {gt_img_info[img_id]['width']}x{gt_img_info[img_id]['height']}")
        
        if img_id not in gt_img_ids:
            print(f"Warning: {img_id} not in ground truth, skipping.")
            continue
        
        # Pass ground truth annotations to yolov5_nano.run
        preds = yolov5_nano.run(img_path, OUTPUT_BASE, gt_annotations[img_id])
        pred_dict.update(format_predictions_yolov5(preds, img_id, gt_img_info))
    
    if pred_dict:
        sample_img_id = next(iter(pred_dict))
        print(f"\nSample Predictions for ID {sample_img_id}: {pred_dict[sample_img_id]}")
        gt_anns = gt_annotations[sample_img_id]
        print(f"Ground Truth for ID {sample_img_id}: {gt_anns}")
    
    metrics = evaluate_predictions(GT_FILE, pred_dict)
    print("\nYOLOv5-Nano Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    run_inference()
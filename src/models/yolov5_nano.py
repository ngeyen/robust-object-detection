import torch
import os
import cv2
import numpy as np

MODEL_PATH = "models/pretrained/yolov5n.pt"
VEHICLE_CLASS_IDS = [2, 3, 4, 5, 6, 7, 8, 9]

def load_model():
    if not os.path.exists(MODEL_PATH):
        torch.hub.download_url_to_file("https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt", MODEL_PATH)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=False)
    model.conf = 0.05  # Lower confidence threshold
    model.iou = 0.3   # Lower NMS IoU threshold
    return model

def run(image_path, output_dir="experiments/baseline_tests/yolov5_nano", gt_annotations=None):
    model = load_model()
    results = model(image_path)
    preds = results.xyxy[0]
    vehicle_mask = torch.isin(preds[:, 5], torch.tensor(VEHICLE_CLASS_IDS))
    vehicle_preds = preds[vehicle_mask]
    
    print("Vehicle Bounding Box Tensor:")
    print(vehicle_preds)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Draw predicted boxes (green)
    for pred in vehicle_preds:
        x_min, y_min, x_max, y_max, conf, cls = pred.tolist()
        # Scale predictions to original image size (already in 640x640, scale back later in evaluation)
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        label = f"Pred Class {int(cls)}: {conf:.2f}"
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green for predictions
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw ground truth boxes (red) if provided
    if gt_annotations:
        for ann in gt_annotations:
            x_min, y_min, x_max, y_max = map(int, ann['bbox'])
            label = f"GT Class {ann['category_id']}"
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red for ground truth
            cv2.putText(img, label, (x_min, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Image with predicted and ground truth bounding boxes saved to: {output_path}")
    
    return vehicle_preds

if __name__ == "__main__":
    sample_image = "data/processed/coco_occluded_vehicles/000000045596.jpg"
    # For standalone testing, load sample GT (remove in production)
    with open("data/annotations/coco_occluded_vehicles.json", 'r') as f:
        gt_data = json.load(f)
    gt_anns = [ann for ann in gt_data['annotations'] if ann['image_id'] == 45596]
    gt_anns_converted = [{'bbox': [ann['bbox'][0], ann['bbox'][1], 
                                   ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]], 
                          'category_id': ann['category_id']} for ann in gt_anns]
    run(sample_image, gt_annotations=gt_anns_converted)
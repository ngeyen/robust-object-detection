import tensorflow as tf
import cv2
import numpy as np
import os

MODEL_PATH = "models/pretrained/efficientdet-tensorflow2-lite0-detection-v1"
VEHICLE_CLASS_IDS =  [2, 3, 4, 5, 6, 7, 8, 9]  # Adjusted to include 0 and 0-indexed COCO vehicles

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please download and extract it.")
    model = tf.saved_model.load(MODEL_PATH)
    return model

def nms(boxes, scores, iou_threshold=0.5, max_boxes=10):
    if len(boxes) == 0:
        return np.array([]), np.array([])
    
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    
    areas = (x_max - x_min) * (y_max - y_min)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0 and len(keep) < max_boxes:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x_min[i], x_min[order[1:]])
        yy1 = np.maximum(y_min[i], y_min[order[1:]])
        xx2 = np.minimum(x_max[i], x_max[order[1:]])
        yy2 = np.minimum(y_max[i], y_max[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep], scores[keep]

def run(image_path, output_dir="experiments/baseline_tests/efficientdet_lite0", gt_annotations=None):
    model = load_model()
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    orig_height, orig_width = img.shape[:2]
    input_tensor = cv2.resize(img, (320, 320))  # EfficientDet-Lite0 uses 320x320
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.uint8)  # SavedModel expects UINT8
    
    # Run inference
    detections = model(input_tensor)
    
    # Debug: Print structure
    print("Detections type:", type(detections))
    if isinstance(detections, tuple):
        print("Detections length:", len(detections))
        for i, tensor in enumerate(detections):
            print(f"Tensor {i} shape: {tensor.shape}")
    
    # Assume tuple output: (boxes, classes, scores, num_detections)
    if isinstance(detections, tuple) and len(detections) >= 4:
        boxes = detections[0][0].numpy()  # [y_min, x_min, y_max, x_max] in 320x320 space
        classes = detections[1][0].numpy().astype(int)
        scores = detections[2][0].numpy()
        num_detections = int(detections[3][0].numpy())
    else:
        raise ValueError("Unexpected detections format. Check model output signature.")

    # Debug: Print raw predictions
    print("Raw Predictions (top 5):")
    for i in range(min(5, num_detections)):
        print(f"Box: {boxes[i]}, Class: {classes[i]}, Score: {scores[i]}")
    
    vehicle_preds = []
    conf_threshold = 0.1
    for i in range(num_detections):
        if scores[i] >= conf_threshold and classes[i] in VEHICLE_CLASS_IDS:
            y_min, x_min, y_max, x_max = boxes[i]
            # Scale from 320x320 to original image size
            x_min_scaled = x_min * orig_width / 320
            y_min_scaled = y_min * orig_height / 320
            x_max_scaled = x_max * orig_width / 320
            y_max_scaled = y_max * orig_height / 320
            vehicle_preds.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled, scores[i], classes[i]])
    
    if vehicle_preds:
        preds_array = np.array(vehicle_preds)
        boxes_to_nms = preds_array[:, :4]
        scores_to_nms = preds_array[:, 4]
        boxes_kept, scores_kept = nms(boxes_to_nms, scores_to_nms, iou_threshold=0.5, max_boxes=10)
        vehicle_preds = []
        for i, box in enumerate(boxes_kept):
            vehicle_preds.append([box[0], box[1], box[2], box[3], scores_kept[i], preds_array[i, 5]])
    
    print("Vehicle Bounding Box Tensor (after NMS):")
    print(np.array(vehicle_preds) if vehicle_preds else "No detections")
    
    for pred in vehicle_preds:
        x_min, y_min, x_max, y_max, conf, cls = map(float, pred)
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        label = f"Pred Class {int(cls)}: {conf:.2f}"
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if gt_annotations:
        for ann in gt_annotations:
            x_min, y_min, x_max, y_max = map(int, ann['bbox'])
            label = f"GT Class {ann['category_id']}"
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(img, label, (x_min, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Image with predicted and ground truth bounding boxes saved to: {output_path}")
    
    return np.array(vehicle_preds) if vehicle_preds else np.array([])

if __name__ == "__main__":
    sample_image = "data/processed/coco_occluded_vehicles/000000045596.jpg"
    with open("data/annotations/coco_occluded_vehicles.json", 'r') as f:
        gt_data = json.load(f)
    gt_anns = [ann for ann in gt_data['annotations'] if ann['image_id'] == 45596]
    gt_anns_converted = [{'bbox': [ann['bbox'][0], ann['bbox'][1], 
                                   ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]], 
                          'category_id': ann['category_id']} for ann in gt_anns]
    run(sample_image, gt_annotations=gt_anns_converted)
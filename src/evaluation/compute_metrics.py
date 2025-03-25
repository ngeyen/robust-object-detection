import json
import numpy as np
from collections import defaultdict

VEHICLE_CAT_IDS = [2, 3, 4, 5, 6, 7, 8, 9]

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_predictions(gt_file, pred_dict, iou_threshold=0.5):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    gt_images = {img['id']: img for img in gt_data['images']}
    gt_annotations = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_annotations[ann['image_id']].append({
            'bbox': [ann['bbox'][0], ann['bbox'][1], 
                     ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]],
            'category_id': ann['category_id'],
            'used': False
        })

    all_predictions = []
    for image_id, preds in pred_dict.items():
        for pred in preds:
            all_predictions.append({
                'image_id': image_id,
                'bbox': pred['bbox'],
                'score': pred['confidence'],
                'category_id': int(pred['class_id'])
            })

    all_predictions.sort(key=lambda x: x['score'], reverse=True)

    tp, fp = [], []
    gt_matched = set()
    
    for pred in all_predictions:
        img_id = pred['image_id']
        if img_id not in gt_annotations:
            fp.append(1)
            tp.append(0)
            continue
        
        gt_anns = gt_annotations[img_id]
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_anns):
            if gt['used']:
                continue
            # Match any vehicle category
            if pred['category_id'] not in VEHICLE_CAT_IDS or gt['category_id'] not in VEHICLE_CAT_IDS:
                continue
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            gt_anns[best_gt_idx]['used'] = True
            gt_matched.add((img_id, best_gt_idx))
        else:
            tp.append(0)
            fp.append(1)

    fn = sum(len([gt for gt in anns if not gt['used']]) for anns in gt_annotations.values())
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / (fn + tp_cum + 1e-6)

    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0

    mAP = ap
    total_preds = len(tp)
    total_gt = sum(len(anns) for anns in gt_annotations.values())
    precision_final = tp_cum[-1] / total_preds if total_preds > 0 else 0
    recall_final = tp_cum[-1] / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision_final * recall_final) / (precision_final + recall_final + 1e-6)

    return {
        'mAP@0.5': mAP,
        'Precision': precision_final,
        'Recall': recall_final,
        'F1': f1,
        'TP': int(tp_cum[-1]),
        'FP': int(fp_cum[-1]),
        'FN': fn
    }

def format_predictions_yolov5(preds_tensor, image_id, gt_img_info):
    img_info = gt_img_info[image_id]
    orig_width, orig_height = img_info['width'], img_info['height']
    scale_x, scale_y = orig_width / 640, orig_height / 640
    
    preds = []
    for pred in preds_tensor:
        x_min, y_min, x_max, y_max, conf, cls = pred.tolist()
        x_min_scaled = x_min * scale_x
        y_min_scaled = y_min * scale_y
        x_max_scaled = x_max * scale_x
        y_max_scaled = y_max * scale_y
        preds.append({
            'bbox': [x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled],
            'confidence': conf,
            'class_id': cls
        })
    return {image_id: preds}

if __name__ == "__main__":
    sample_preds_tensor = torch.tensor([
        [1.14, 99.00, 93.11, 188.77, 0.4247, 7.0],
        [484.89, 168.63, 576.83, 258.10, 0.4104, 2.0],
        [118.17, 166.55, 165.89, 207.19, 0.3767, 2.0]
    ])
    with open("data/annotations/coco_occluded_vehicles.json", 'r') as f:
        gt_data = json.load(f)
    gt_img_info = {img['id']: img for img in gt_data['images']}
    pred_dict = format_predictions_yolov5(sample_preds_tensor, 284698, gt_img_info)
    metrics = evaluate_predictions("data/annotations/coco_occluded_vehicles.json", pred_dict)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
import torch
import os

MODEL_PATH = "models/pretrained/yolov5n.pt"

def load_model():
    """Load pre-trained YOLOv5-Nano."""
    if not os.path.exists(MODEL_PATH):
        # Download from YOLOv5 GitHub
        torch.hub.download_url_to_file("https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt", MODEL_PATH)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    return model

def run(image_path):
    """Run inference on a single image."""
    model = load_model()
    results = model(image_path)
    return results.xyxy[0]  # Bounding boxes in [x_min, y_min, x_max, y_max, conf, class]
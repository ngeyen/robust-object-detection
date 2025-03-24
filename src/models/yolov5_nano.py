import torch
import os

MODEL_PATH = "models/pretrained/yolov5n.pt"
SAVE_DIR = "experiments/baseline_tests/yolov5_nano"

def load_model():
    """Load pre-trained YOLOv5-Nano."""
    if not os.path.exists(MODEL_PATH):
        # Download from YOLOv5 GitHub
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.hub.download_url_to_file("https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt", MODEL_PATH)
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    return model

def run(image_path, save_dir=SAVE_DIR):
    """Run inference on a single image and save the results."""
    model = load_model()
    results = model(image_path)
    results.save(save_dir=save_dir)
    return results.xyxy[0]  # Bounding boxes in [x_min, y_min, x_max, y_max, conf, class]

if __name__ == "__main__":
    # Create the save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Define the image path
    image_path = "data/result/8a2.jpg"

    # Run inference and save the results
    results_tensor = run(image_path)
    print("Bounding Box Tensor:")
    print(results_tensor)

    print(f"Image with bounding boxes saved to: {SAVE_DIR}")
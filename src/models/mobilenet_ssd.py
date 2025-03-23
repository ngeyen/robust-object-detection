import tensorflow as tf
import os

MODEL_PATH = "models/pretrained/ssd_mobilenet_v2"

def load_model():
    """Load pre-trained MobileNet-SSD V2."""
    if not os.path.exists(MODEL_PATH):
        # Download from TensorFlow Model Zoo
        url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
        tf.keras.utils.get_file("ssd_mobilenet_v2.tar.gz", url, untar=True, cache_dir="models/pretrained")
    model = tf.saved_model.load(MODEL_PATH)
    return model

def run(image_path):
    """Run inference on a single image."""
    model = load_model()
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [300, 300])  # Model expects 300x300
    image = tf.expand_dims(image, 0)  # Batch dimension
    detections = model(image)
    return detections
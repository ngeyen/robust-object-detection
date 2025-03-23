import tensorflow as tf
import os

MODEL_PATH = "models/pretrained/ssdlite_mobiledet"

def load_model():
    """Load pre-trained SSDLite-MobileDet."""
    if not os.path.exists(MODEL_PATH):
        # Download from TensorFlow Hub
        url = "https://tfhub.dev/tensorflow/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19/1"
        tf.keras.utils.get_file("ssdlite_mobiledet.tar.gz", url, untar=True, cache_dir="models/pretrained")
    model = tf.saved_model.load(MODEL_PATH)
    return model

def run(image_path):
    """Run inference on a single image."""
    model = load_model()
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [320, 320])  # Model expects 320x320
    image = tf.expand_dims(image, 0)
    detections = model(image)
    return detections
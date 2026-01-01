import tensorflow as tf
import cv2
import numpy as np
from .config import IMG_HEIGHT, IMG_WIDTH

def preprocess_image(image):
    """
    Prepares an image for the MobileNetV2 model.
    1. Resizes to (224, 224).
    2. Converts to float32.
    3. Normalizes using tf.keras.applications.mobilenet_v2.preprocess_input.
    """
    try:
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def prepare_for_inference(image):
    """
    Expands dims to create a batch of 1.
    """
    processed = preprocess_image(image)
    if processed is not None:
        return np.expand_dims(processed, axis=0)
    return None

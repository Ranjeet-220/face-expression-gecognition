import os

# Data Config
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
BATCH_SIZE = 32
EPOCHS = 20

# Model Config
NUM_CLASSES = 7
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.h5")

# Face Detection Config
# We will resolve the cascade path dynamically in face_detector.py to avoid heavy imports here
# or we can assume a standard accessible path if copied locally.
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'

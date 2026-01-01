import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# Config
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def generate_data(base_dir, num_samples=10):
    for split in ['train', 'validation']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
            
        print(f"Generating {split} data...")
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
                
            for i in tqdm(range(num_samples), desc=f"{split}/{class_name}"):
                # Random noise image
                img = np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                
                # Save
                filename = os.path.join(class_dir, f"{class_name}_{i}.jpg")
                cv2.imwrite(filename, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--samples", type=int, default=10, help="Images per class per split")
    args = parser.parse_args()
    
    generate_data(args.data_dir, args.samples)

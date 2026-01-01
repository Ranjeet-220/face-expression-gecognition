import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse

# Config
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def process_fer2013(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    
    # FER2013 Usage column: Training, PublicTest (Validation), PrivateTest (Test)
    # We will map Training -> train, PublicTest -> validation
    
    usage_map = {
        'Training': 'train',
        'PublicTest': 'validation',
        'PrivateTest': 'validation' # Adding PrivateTest to validation for simplicity in this project scope
    }

    for usage, split_name in usage_map.items():
        subset = df[df['Usage'] == usage]
        if subset.empty:
            continue
            
        print(f"Processing {usage} -> {split_name} ({len(subset)} images)...")
        
        for index, row in tqdm(subset.iterrows(), total=subset.shape[0]):
            emotion_label = row['emotion']
            pixels = row['pixels']
            
            emotion_name = CLASS_NAMES.get(emotion_label, "Unknown")
            
            # Create directory
            save_dir = os.path.join(output_dir, split_name, emotion_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Convert pixels to image
            # pixels are space separated string
            face = np.array(pixels.split(' '), dtype='uint8').reshape(48, 48)
            
            # Resize > 224x224
            face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
            
            # Save
            image_path = os.path.join(save_dir, f"{usage}_{index}.jpg")
            cv2.imwrite(image_path, face_resized)

    print(f"Processing complete. Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="r:\\Project\\data\\fer2013.csv", help="Path to fer2013.csv")
    parser.add_argument("--output_dir", type=str, default="r:\\Project\\data", help="Output directory")
    args = parser.parse_args()
    
    process_fer2013(args.csv_path, args.output_dir)

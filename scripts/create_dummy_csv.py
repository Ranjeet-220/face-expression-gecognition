import pandas as pd
import numpy as np
import argparse

# Config
NUM_SAMPLES = 50 # Total lines to generate

def create_dummy_csv(output_path):
    print(f"Generating dummy FER2013 CSV at {output_path}...")
    
    data = []
    usages = ['Training', 'PublicTest', 'PrivateTest']
    
    for i in range(NUM_SAMPLES):
        emotion = np.random.randint(0, 7)
        # Generate 48x48 = 2304 pixels as space separated string
        pixels_list = np.random.randint(0, 255, 2304)
        pixels_str = " ".join(map(str, pixels_list))
        usage = np.random.choice(usages)
        
        data.append([emotion, pixels_str, usage])
        
    df = pd.DataFrame(data, columns=['emotion', 'pixels', 'Usage'])
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="r:\\Project\\data\\fer2013.csv")
    args = parser.parse_args()
    
    create_dummy_csv(args.output_path)

import kagglehub
import os

print("Starting dataset download...")
# Download latest version
path = kagglehub.dataset_download("jonathanoheix/face-expression-recognition-dataset")

print(f"Dataset downloaded to: {path}")

# Check content
if os.path.exists(path):
    print("Files in download directory:")
    print(os.listdir(path))

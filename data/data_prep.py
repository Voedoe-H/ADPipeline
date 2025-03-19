import kagglehub
import os

download_path = os.path.join(os.path.dirname(__file__),'raw')
os.makedirs(download_path, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("ipythonx/mvtec-ad",path=download_path)

print("Path to dataset files:", path)
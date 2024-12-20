"""
Downloads and extracts an image dataset of pizzas, steaks, and sushi from an
online source if it does not already exist in the specified local directory.
"""
import os
import zipfile
from pathlib import Path

import requests

# Setup path to a data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
zip_path = os.path.join(data_path, "pizza_steak_sushi.zip")

print(type(image_path), image_path)
print(type(zip_path), zip_path)

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"\n{image_path} directory already exists... skipping download")
else:
    print(f"\n{image_path} does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak and sushi data
    with open(zip_path, "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("Unzipping pizza, steak and sushi data...")
        zip_ref.extractall(image_path)

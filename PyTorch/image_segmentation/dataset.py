import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # List all the files that are in that folder (listdir)
        self.images = os.listdir(image_dir)

    def __len__(self):
        """
        Length of dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Get item by image index
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        # image = np.array(Image.open(img_path))  # Try this if it didn't work.
        # PIL: numpy array, grayscale mask.
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # 0.0, 255.0
        # Look for the mask where it's equal to 255, and change it to 1
        # bc last activation is sigmoid, indicating probability it's a white pixel,
        # correct for label, convert to 1.
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            # Data Augmentation
            augmentations = self.transform(image=image, mask=mask)
            # Get the augmented image from the dictionary with the key "image"
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

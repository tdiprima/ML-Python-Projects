from torch.utils.data import Dataset  # all PyTorch datasets must inherit from this base class
import cv2  # image handling


class SegmentationDataset(Dataset):
    """
    Each PyTorch dataset is required to inherit from Dataset class
    and should have a __len__ and a __getitem__
    """

    def __init__(self, image_paths, mask_paths, transforms):
        """
        Initializer constructor
        image_paths: list of image paths
        mask_paths: corresponding ground-truth masks
        transforms: set of transformations
        """
        # store the image and mask filepaths, and augmentation transforms
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        # Return the total number of image paths in our dataset (total samples)
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Take an index as input and return the corresponding sample from the dataset.
        # Simply grab the image path at the "idx" index in our list of input image paths
        image_path = self.image_paths[idx]

        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], 0)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)

        # return a tuple of the image and its mask
        return (image, mask)

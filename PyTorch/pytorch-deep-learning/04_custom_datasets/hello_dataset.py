import os
import random
from pathlib import Path

import torch
from PIL import Image
# from PIL import ImageShow
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from display_rand_imgs import display_random_images

print("Torch version:", torch.__version__)

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
print("cpu count", os.cpu_count())

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
image_path_list = list(image_path.glob("*/*/*.jpg"))

if not image_path.is_dir():
    print("Data folder doesn't exist. Exiting...")
    exit(0)

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# DATA TRANSFORM
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),  # p=probability; 50% of the time
    transforms.ToTensor()
])

# LOAD IMAGE DATA
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,  # a transform for the data
                                  target_transform=None)  # a transform for the label/target

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

class_names = train_data.classes
class_dict = train_data.class_to_idx


def print_stuff():
    print("\nData Len:", len(train_data), len(test_data))
    print("\nTrain Data:\n", train_data)
    print("\nTest Data:\n", test_data)

    print("\nClass Names:\n", class_names)
    print("\nClass Dict:\n", class_dict)

    # print("\nA Sample:\n", train_data[0])

    # Index on the train_data Dataset to get a single image and label
    img, label = train_data[0][0], train_data[0][1]
    # "Image tensor:\n {img}"
    print(f"\nImage shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")


# print_stuff()  # TODO

# DATA SETS => DATA LOADERS
BATCH_SIZE = 1
NUM_WORKERS = os.cpu_count()
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=1,  # How many cpu cores used to load your data
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)


def plot_transformed_images(image_paths: list, transform, n=3, seed=None):
    """
    Selects random images from a path of images and loads/transforms
    them then plots the original vs the transformed version.
    """
    if seed:
        random.seed(seed)

    random_image_paths = random.sample(image_paths, k=n)  # k = number of samples

    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            # Change shape for matplotlib (C, H, W) -> (H, W, C)
            transformed_image = transform(f).permute(1, 2, 0)  # swap order of axes
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

            plt.show()


def get_rand_image():
    # Get all image paths
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # Pick a random image path
    random_image_path = random.choice(image_path_list)

    # Get image class from directory name
    image_class = random_image_path.parent.stem

    return random_image_path, image_class


def show_image_pil():
    random_image_path, image_class = get_rand_image()

    # Open image
    img = Image.open(random_image_path)

    # Print metadata
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")

    # todo: Not all viewers can display the title.
    # img.show(title="Show image with PIL")
    # ImageShow.show(img, title="Show image with PIL")
    img.show()


def show_image_plt():
    # https://www.askpython.com/python/examples/display-images-using-python
    random_image_path, image_class = get_rand_image()
    plt.title(f"Show {image_class} with matplotlib")
    plt.xlabel("X pixel scaling")
    plt.ylabel("Y pixel scaling")

    image = mpimg.imread(random_image_path)
    plt.imshow(image)
    plt.show()


# TODO:
# show_image_pil()
# show_image_plt()
# plot_transformed_images(image_paths=image_path_list,
#                         transform=data_transform,
#                         n=3,
#                         seed=None)
# Display random images from the ImageFolder created Dataset
display_random_images(train_data,
                      n=3,
                      classes=class_names,
                      seed=None)

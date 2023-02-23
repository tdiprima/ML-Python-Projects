import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from skimage.color import rgb2gray
from skimage.io import imread


def growcut(image, markers, num_iters=5):
    # Initialize the foreground and background labels
    labels = np.zeros_like(image)
    labels[markers == 1] = 1
    labels[markers == 2] = 2

    # Define the foreground and background colors
    fg_color = np.mean(image[markers == 1])
    bg_color = np.mean(image[markers == 2])

    print(fg_color, bg_color)

    # Perform the GrowCut iterations
    for i in range(num_iters):
        # Compute the region means
        fg_mean = np.mean(image[labels == 1])
        bg_mean = np.mean(image[labels == 2])

        # Update the labels
        labels[(image - fg_mean) ** 2 < (image - bg_mean) ** 2] = 1
        labels[(image - fg_mean) ** 2 >= (image - bg_mean) ** 2] = 2

    # Label the regions and return the segmentation
    num_regions, regions = label(labels == 1)
    return regions


# IMPLEMENT GROWCUT
image_name = 'input.png'

# Load the input image and convert it to grayscale
try:
    image = imread(image_name)
    print(image.shape)

    # just use the first 3 channels instead of the 4 you have
    # image = rgb2gray(image[..., 0:3])

    img = rgb2gray(image)
    print(image.shape)

    # image = image.reshape(512, 512, 3)
except Exception as ex:
    print("rgb2gray:", ex)
    exit(1)

# Define the markers
markers = np.zeros_like(image)
markers[50:100, 50:100] = 1  # foreground
markers[150:200, 150:200] = 2  # background

# Perform the segmentation using the GrowCut algorithm
seg = growcut(image, markers)

# Visualize the segmentation
try:
    plt.imshow(seg)  # plt.imshow expects image data as [height, width, 3]
    plt.show()
except Exception as ex:
    print("Sigh. Wtf...", ex)
    exit(1)

"""
Algorithm returns 1 number. Something's not right.
"""
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


# Load the input image
lulu = imread('../images/dragonball.jpg')  # Use this one, only.
print("img shape:", lulu.shape)

# Convert it to grayscale
image = rgb2gray(lulu)
print("bw shape:", lulu.shape)

# Define the markers
markers = np.zeros_like(image)
markers[50:100, 50:100] = 1  # foreground
markers[150:200, 150:200] = 2  # background

# Perform the segmentation using the GrowCut algorithm
seg = growcut(image, markers)
print("seg", type(seg), seg)  # <class 'int'> 309

# Visualize the segmentation
try:
    plt.imshow(seg)  # plt.imshow expects image data as [height, width, 3]
except Exception as ex:
    print("imshow:", ex)
    exit(1)

plt.show()

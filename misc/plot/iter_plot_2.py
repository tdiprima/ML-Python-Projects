"""
Creates a matplotlib figure with a grid of subplots to display multiple 2D images in a window.
TODO: Scrollbar
https://stackoverflow.com/questions/42622146/scrollbar-on-matplotlib-showing-page
"""
import matplotlib.pyplot as plt
import numpy as np

# assume you have a list of image data, where each image is a 2D numpy array
image_list = [np.random.rand(10, 10) for _ in range(50)]

# calculate the number of rows and columns needed to display all the images
num_images = len(image_list)
num_rows = int(np.ceil(np.sqrt(num_images)))
num_cols = int(np.ceil(num_images / num_rows))

# create a figure object and an array of axes objects to plot the images
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

# iterate through the list of images and plot each image in a separate axis object
for i, image in enumerate(image_list):
    row_idx = i // num_cols
    col_idx = i % num_cols
    axs[row_idx, col_idx].imshow(image, cmap='gray')

# set the title and axis labels for the figure
fig.suptitle('List of Images')
fig.tight_layout()
plt.show()

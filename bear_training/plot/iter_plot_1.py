import matplotlib.pyplot as plt
import numpy as np

# assume you have a list of image data, where each image is a 2D numpy array
image_list = [np.random.rand(10, 10), np.random.rand(10, 10), np.random.rand(10, 10)]

# create a figure object and an array of axes objects to plot the images
fig, axs = plt.subplots(1, len(image_list), figsize=(10, 5))

# iterate through the list of images and plot each image in a separate axis object
for i, image in enumerate(image_list):
    axs[i].imshow(image)

# set the title and axis labels for the figure
fig.suptitle('List of Images')
fig.tight_layout()
plt.show()

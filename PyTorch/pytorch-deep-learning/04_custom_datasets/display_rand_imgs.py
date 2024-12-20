"""
Displays a random selection of images from a provided dataset using matplotlib,
with the option to label by class and display image dimensions.
"""
import random
from typing import List

import torch
from matplotlib import pyplot as plt


def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    """
    Take in a dataset, and display random resized images.
    """

    # Adjust display if n is too high
    if n > 10:
        print(f"Expected n=10; got n={n}")
        n = 10
        display_shape = False
        # print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # Set the seed
    if seed:
        random.seed(seed)

    # Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # Setup plot
    plt.figure(figsize=(16, 8))

    # Loop through random indexes and plot them with matplotlib
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # Adjust tensor dimensions for plotting
        # [color_channels, height, width] -> [height, width, color_channels]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)

    plt.show()

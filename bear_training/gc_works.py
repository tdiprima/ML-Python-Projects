import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label


def growcut(image, seed, num_iter=10):
    # Initialize the labels with the seed
    labels = seed.copy()
    # Get the unique label values
    label_vals = np.unique(seed)
    # Get the number of labels
    num_labels = len(label_vals)

    # Iterate for the given number of iterations
    for i in range(num_iter):
        # Loop over each label
        for j in range(num_labels):
            # Get the pixels with the current label
            label_mask = (labels == label_vals[j])
            # Get the pixels in the background
            bg_mask = (labels == 0)
            # Get the pixels in the foreground
            fg_mask = (label_mask & ~bg_mask)

            # Compute the mean of the foreground and background pixels
            bg_mean = np.mean(image[bg_mask])
            fg_mean = np.mean(image[fg_mask])

            # Update the labels based on the difference in mean intensity
            if fg_mean > bg_mean:
                labels[label_mask] = label_vals[j]
            else:
                labels[fg_mask] = 0

        # Label connected components of the foreground
        _, num_fg_labels = label(labels != 0)
        # Remove small connected components
        for j in range(1, num_fg_labels + 1):
            fg_mask = (labels == j)
            if np.sum(fg_mask) < 10:
                labels[fg_mask] = 0

    return labels


# Generate a random image
np.random.seed(0)
image = np.random.rand(100, 100)

# Generate a seed image
seed = np.zeros_like(image, dtype=np.int32)
seed[40:60, 40:60] = 1
seed[70:80, 10:20] = 2

# Run the GrowCut algorithm
labels = growcut(image, seed)

# Display the results
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input Image')
ax[1].imshow(seed, cmap='tab20b')
ax[1].set_title('Seed Image')
ax[2].imshow(labels, cmap='tab20b')
ax[2].set_title('Segmentation Result')
plt.show()

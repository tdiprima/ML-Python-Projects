import cv2
import numpy as np


def growcut(image, seeds):
    """
    Function for GrowCut segmentation.

    image: np.array
        The input image to be segmented.
    seeds: np.array
        Seeds for segmentation, where 0 indicates background, 1 indicates the object, and 0.5 for unknown.
    """
    height, width = image.shape[:2]
    intensity = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
    state = np.zeros((height, width, 2))
    strength = np.zeros((height, width))
    mask = np.zeros((height, width))
    state[seeds == 0] = 0, 0
    state[seeds == 1] = 1, 1
    strength[seeds == 0] = 1
    strength[seeds == 1] = 1
    while True:
        mask_prev = mask
        for i in range(height):
            for j in range(width):
                for I in [-1, 0, 1]:
                    for J in [-1, 0, 1]:
                        if I != 0 or J != 0:
                            y = i + I
                            x = j + J
                            if y >= 0 and y < height and x >= 0 and x < width:
                                g = np.exp(-(intensity[i, j] - intensity[y, x]) ** 2)
                                attack = g * state[y, x, 0], g * state[y, x, 1]
                                if attack[0] > state[i, j, 0] and attack[1] > strength[i, j]:
                                    strength[i, j] = attack[1]
                                    state[i, j] = attack[0], g
        mask = np.argmax(state, axis=-1)
        if np.all(mask == mask_prev):
            break
    return mask


# Load your image
image = cv2.imread('your_image_path.jpg')

# Load your seeds
seeds = cv2.imread('your_seeds_path.jpg', cv2.IMREAD_GRAYSCALE)

# Run the growcut segmentation
segmented_image = growcut(image, seeds)

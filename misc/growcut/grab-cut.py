"""
Extracts the foreground of an image using the GrabCut algorithm from OpenCV, and then segments the foreground from the
background, displaying the resulting segmented image with a color-bar.
https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Image is loaded with imread
# image = cv2.imread('lights_02.png')
# image = cv2.imread('match.png')
# image = cv2.imread('retina-noise.png')
image = cv2.imread('dragonball.jpg')

# Create a simple mask image, similar to the loaded image, with the shape and return type
mask = np.zeros(image.shape[:2], np.uint8)

"""
Specify the background and foreground model using numpy.
The array is constructed of 1 row and 65 columns, and all array elements are 0.
Data type for the array is np.float64 (default)
"""
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

"""
Define the Region of Interest (ROI) as the coordinates of the rectangle
where the values are entered as
(startingPoint_x, startingPoint_y, width, height)
These coordinates are according to the input image.
"""
rectangle = (20, 100, 150, 150)

"""
Apply the grabcut algorithm with appropriate values as parameters,
number of iterations = 3.
cv2.GC_INIT_WITH_RECT is used because the rectangle mode is used.
"""
cv2.grabCut(image, mask, rectangle,
            backgroundModel, foregroundModel,
            3, cv2.GC_INIT_WITH_RECT)

"""
In the new mask image, pixels will be marked with four flags.
Four flags denote the background / foreground.
Mask is changed, all the 0 and 2 pixels are converted to the background.
Mask is changed, all the 1 and 3 pixels are now part of the foreground.
The return type is also mentioned, this gives us the final mask.
"""
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# The final mask is multiplied with the input image to give the segmented image.
image = image * mask2[:, :, np.newaxis]

# Output segmented image with color-bar.
plt.imshow(image)
plt.colorbar()
plt.show()

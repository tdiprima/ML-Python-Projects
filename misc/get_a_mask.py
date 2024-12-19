"""
Atención: Conversion to mask took a long time for some reason
Reads an image, converts it to a binary mask based on a threshold, and then displays the mask using matplotlib.
"""

import torch
from PIL import Image
from tensorflow.keras.preprocessing import image

# Assume your tensor is called tensor
test_image = Image.open("Formula1.jpg").convert("L")
img = image.img_to_array(test_image)
img = img.reshape((1,) + img.shape)
tensor = torch.from_numpy(img).float()

# Now, let's make the binary mask
tensor = tensor.mean(dim=1, keepdim=True)
threshold = 0.5  # Choose a threshold
mask = tensor > threshold

# Permute to (N, C, H, W)
tensor = tensor.permute(0, 3, 1, 2)

# Do the same for the mask
mask = mask.permute(0, 3, 1, 2)

# They're both shape [1, 3, 1, 256]

import matplotlib.pyplot as plt

# Convert the tensor mask to a numpy array and squeeze it to 2D
mask_np = mask.numpy().squeeze()

# Using matplotlib to display the mask
plt.imshow(mask_np, cmap='gray')
plt.show()


exit(0)

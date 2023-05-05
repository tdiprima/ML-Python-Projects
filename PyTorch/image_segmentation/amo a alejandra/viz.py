import matplotlib.pyplot as plt

from PIL import Image

im = Image.open("data/train/images/7cc53fe88b.png")
plt.subplot(2, 2, 1)
plt.imshow(im)

im = Image.open("data/train/masks/7cc53fe88b.png")
plt.subplot(2, 2, 2)
plt.imshow(im)

im = Image.open("data/train/images/9ca520f895.png")
plt.subplot(2, 2, 3)
plt.imshow(im)

im = Image.open("data/train/masks/9ca520f895.png")
plt.subplot(2, 2, 4)
plt.imshow(im)

plt.show()

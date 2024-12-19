# This script opens an image, applies various random augmentations like rotation, zoom, flip etc. using Keras ImageDataGenerator and displays the augmented images.
"""
https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/
"""
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
from tensorflow.keras.preprocessing import image

test_image = Image.open("formula_one.png")
img = image.img_to_array(test_image)  # (289, 512, 4)
img = img.reshape((1,) + img.shape)  # (1, 289, 512, 4)
# img = img.unsqueeze(1)  # can't unsqueeze a numpy array

# ImageDataGenerator rotation
# datagen = ImageDataGenerator(rotation_range=30, fill_mode='nearest')

# Let's see what we're doing in img-seg-steps.py
datagen = ImageDataGenerator(rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True)

# iterator
aug_iter = datagen.flow(img, batch_size=1)

# generate samples and plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(7, 7))

# generate batch of images
for i in range(3):
    # convert to unsigned integers
    im = next(aug_iter)[0].astype('uint8')
    # plot image
    ax[i].imshow(im)
    ax[i].axis('off')

plt.show()

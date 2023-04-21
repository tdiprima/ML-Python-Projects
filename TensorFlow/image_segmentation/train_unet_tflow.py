#!/usr/bin/env python
"""
https://stackoverflow.com/questions/50304156/tensorflow-allocation-memory-allocation-of-38535168-exceeds-10-of-system-memor
https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/42121886#42121886
"""
# If you don't leave it like this, you get errors:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import PIL
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras import Sequential,Model
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import os
from tensorflow.keras.backend import flatten
import tensorflow.keras.backend as K

home_dir = os.path.expanduser('~')

# TODO:
grinder = "/projects/image_segmentation/data"
local = "/Documents/data"

# Prepare data generator (no augmentation)
data_dir = home_dir + grinder + '/train/'
mask_dir = home_dir + grinder + '/train_masks/'

all_images = os.listdir(data_dir)

# TODO:
# to_train = 1  # all images
to_train = 0.1  # some of the images

total_train_images = all_images[:int(len(all_images) * to_train)]

WIDTH = 512  # actual : 1918//1920 divisive by 64
HEIGHT = 512  # actual : 1280

# TODO:
# Keep small-ish. "Allocation of N exceeds 10% of free system memory."
# BATCH_SIZE = 8
BATCH_SIZE = 16

# TODO: Set the number of worker nodes
# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 4

# split train set and test set
train_images, validation_images = train_test_split(total_train_images, train_size=0.8, test_size=0.2, random_state=0)


def data_gen_small(data_dir, mask_dir, images, batch_size, dims):
    """
    Generator that we will use to read the data from the directory
    data_dir: images location
    mask_dir: masks location
    images: filenames of the images we want to generate batches from
    batch_size: number of samples that will be propagated through the network
    dims: dimensions in which we want to rescale our images, tuple
    """
    while True:
        ix = np.random.choice(np.arange(len(images)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            original_img = load_img(data_dir + images[i])
            resized_img = original_img.resize(dims)
            array_img = img_to_array(resized_img) / 255
            imgs.append(array_img)

            # masks
            original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')
            resized_mask = original_mask.resize(dims)
            array_mask = img_to_array(resized_mask) / 255
            labels.append(array_mask[:, :, 0])

        imgs = np.array(imgs)
        labels = np.array(labels)
        # function returns a generator
        yield imgs, labels.reshape(-1, dims[0], dims[1], 1)


def data_gen_aug(data_dir, mask_dir, images, batch_size, dims):
    """
    Generator that we will use to read the data from the directory with random augmentation
    data_dir: images location
    mask_dir: masks location
    images: filenames of the images we want to generate batches from
    batch_size: number of samples that will be propagated through the network
    dims: dimensions in which we want to rescale our images, tuple
    """
    while True:
        ix = np.random.choice(np.arange(len(images)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # read images and masks
            original_img = load_img(data_dir + images[i])
            original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')

            # transform into ideal sizes
            resized_img = original_img.resize(dims)
            resized_mask = original_mask.resize(dims)

            # add random augmentation > here we only flip horizontally
            if np.random.random() < 0.5:
                # TODO: Deprecated; use Transpose.FLIP_LEFT_RIGHT.
                resized_img = resized_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                resized_mask = resized_mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

            array_img = img_to_array(resized_img) / 255
            array_mask = img_to_array(resized_mask) / 255

            imgs.append(array_img)
            labels.append(array_mask[:, :, 0])

        imgs = np.array(imgs)
        labels = np.array(labels)
        # function returns a generator
        yield imgs, labels.reshape(-1, dims[0], dims[1], 1)


# generator for train and validation data set
train_gen = data_gen_aug(data_dir, mask_dir, train_images, BATCH_SIZE, (WIDTH, HEIGHT))
val_gen = data_gen_small(data_dir, mask_dir, validation_images, BATCH_SIZE, (WIDTH, HEIGHT))

"""
Set up U-Net model
Define down and up layers that will be used in U-Net model
"""


def down(input_layer, filters, pool=True):
    """
    filters: numbers of filters that convolutional layers will learn from.
    It also determines the number of output filters in the convolution.
    kernel_size: dimensions of the kernel, tuple.
    """
    # Pass the input_layer tensor as input to the Conv2D layer to extract features from it.
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    if pool:
        # MaxPooling2D(pool_size=(3, 3))
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual


def up(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2


# Make a custom U-net implementation.
filters = 64

"""
Input fn returns a KerasTensor
What's wigging me out is: shape=(None, 512, 512, 3). Whaddya mean batch size "None"??
"""
input_layer = Input(shape=[WIDTH, HEIGHT, 3])
layers = [input_layer]
residuals = []

# Down 1
d1, res1 = down(input_layer, filters)
residuals.append(res1)

filters *= 2

# Down 2
d2, res2 = down(d1, filters)
residuals.append(res2)

filters *= 2

# Down 3
d3, res3 = down(d2, filters)
residuals.append(res3)

filters *= 2

# Down 4
d4, res4 = down(d3, filters)
residuals.append(res4)

filters *= 2

# Down 5
d5 = down(d4, filters, pool=False)

# Up 1
up1 = up(d5, residual=residuals[-1], filters=filters / 2)
filters /= 2

# Up 2
up2 = up(up1, residual=residuals[-2], filters=filters / 2)

filters /= 2

# Up 3
up3 = up(up2, residual=residuals[-3], filters=filters / 2)

filters /= 2

# Up 4
up4 = up(up3, residual=residuals[-4], filters=filters / 2)

out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)


# Use Tensorflow to write a custom dice_coefficient metric,
# which is an effective indicator of how much two sets overlap with each other.
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


adam = tf.keras.optimizers.Adam(learning_rate=0.0001)


def try1():
    """
    I'm trying to define NUM_WORKERS
    But apparently, it does not have a num_workers argument in its constructor.
    """
    # Define the distribution strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(num_workers=NUM_WORKERS)

    # Build and compile the model within the distribution strategy scope
    with strategy.scope():
        model = Model(input_layer, out)

    return model


def try2():
    """
    That didn't work, so now using a strategy that uses all available
    devices on a single machine to perform synchronous training
    """
    # Create the strategy
    strategy = tf.distribute.experimental.CollectiveAllReduceStrategy()

    # Define and compile the model within the strategy scope
    with strategy.scope():
        model = Model(input_layer, out)
        model.compile(optimizer=adam, loss=BinaryCrossentropy(), metrics=['accuracy', dice_coef])

    return model


# Original
model = Model(input_layer, out)
model.compile(optimizer=adam, loss=BinaryCrossentropy(), metrics=['accuracy', dice_coef])

print(model.summary())

# Train model
import sys

try:
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = home_dir + grinder + "/unet_cp_full_512batch5/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    # Early stop
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                                  restore_best_weights=False
                                                  )

    # Reduce learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.2,
                                                     patience=3,
                                                     verbose=1,
                                                     min_delta=1e-3, min_lr=1e-6
                                                     )

    # Load weights and continue training
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print("\nLatest:\n", latest)

    if latest is not None:
        model.load_weights(latest)

    # model.compile(optimizer=adam, loss=BinaryCrossentropy(), metrics=['accuracy', dice_coef])

    model.fit(train_gen, callbacks=[cp_callback, early_stop, reduce_lr],
              steps_per_epoch=np.ceil(float(len(train_images)) / float(BATCH_SIZE)),
              # todo: epochs=100,
              epochs=1,
              validation_steps=np.ceil(float(len(validation_images)) / float(BATCH_SIZE)),
              validation_data=val_gen)

    # save the model to a file
    model.save("saved_model.pb")

except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("\nType", exc_type)
    print("\nErr:", exc_obj)
    print("\nLine:", exc_tb.tb_lineno)
    sys.exit(1)

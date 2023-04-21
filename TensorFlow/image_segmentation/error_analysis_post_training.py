#!/usr/bin/env python
"""
https://towardsdatascience.com/image-segmentation-predicting-image-mask-with-carvana-data-32829ca826a0
https://github.com/ZeeTsing/Carvana_challenge.git
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import PIL
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras import Sequential, Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.backend import flatten
import tensorflow.keras.backend as K
from glob import glob

# Set up kernel
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
sess_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7),
                                       allow_soft_placement=True)
sess = tf.compat.v1.Session(config=sess_config)
tf.compat.v1.keras.backend.set_session(sess)

home_dir = os.path.expanduser('~')

# TODO: (grinder)
main_dir = home_dir + "/projects/image_segmentation"

DATA_PATH = main_dir + '/data'

train_dir = DATA_PATH + '/train_flow/'
validate_dir = DATA_PATH + '/validate_flow/'
checkpoint_dir = main_dir + '/output/unet-cp'

# TODO:
# TEST_DATA = os.path.join(DATA_PATH, "/test")
# TRAIN_DATA = os.path.join(DATA_PATH, "/train")
TEST_DATA = os.path.join(DATA_PATH, "/val_images")
TRAIN_DATA = os.path.join(DATA_PATH, "/train_images")

TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "/train_masks")

WIDTH = 512  # actual : 1918//1920 divisive by 64
HEIGHT = 512  # actual : 1280
# BATCH_SIZE = 8
BATCH_SIZE = 16

import cv2
from PIL import Image


def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        ext = 'jpg'
        data_path = TRAIN_DATA
        suffix = ''
    elif "Train_mask" in image_type:
        ext = 'gif'
        data_path = TRAIN_MASKS_DATA
        suffix = '_mask'
    elif "Test" in image_type:
        ext = 'jpg'
        data_path = TEST_DATA
        suffix = ''
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))


def get_image_data(image_id, image_type, **kwargs):
    if 'mask' in image_type:
        img = _get_image_data_pil(image_id, image_type, **kwargs)
    else:
        img = _get_image_data_opencv(image_id, image_type, **kwargs)
    return img


def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _get_image_data_pil(image_id, image_type, return_exif_md=False, return_shape_only=False):
    fname = get_filename(image_id, image_type)
    try:
        img_pil = Image.open(fname)
    except Exception as e:
        assert False, "Failed to read image : %s, %s. Error message: %s" % (image_id, image_type, e)

    if return_shape_only:
        return img_pil.size[::-1] + (len(img_pil.getbands()),)

    img = np.asarray(img_pil)
    assert isinstance(img, np.ndarray), "Open image is not an ndarray. Image id/type : %s, %s" % (image_id, image_type)
    if not return_exif_md:
        return img
    else:
        return img, img_pil._getexif()


# Load trained Unet model

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    if pool:
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

model = Model(input_layer, out)

print(model.summary())

latest = tf.train.latest_checkpoint(checkpoint_dir)
print("Latest:", latest)

if latest is not None:
    model.load_weights(latest)

# Get prediction - Train set

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_date_gen = test_datagen.flow_from_directory(
    train_dir,
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    shuffle=False,
    class_mode=None,
    batch_size=BATCH_SIZE)

filenames = train_date_gen.filenames
nb_samples = len(filenames)

train_ids = [fn.split('\\')[-1][:-4] for fn in filenames]

predict = model.predict(train_date_gen, steps=
np.ceil(nb_samples / BATCH_SIZE))

with open(main_dir + '/output/train-prediction.txt', 'wb') as file:
    file.write(predict)

with open(main_dir + '/output/train-prediction_name.txt', 'w') as file:
    for line in filenames:
        file.write(line + '\n')


# define function to calculate dice coefficient with image_array as input
def dice(y_true, y_pred):
    intersection = sum(map(sum, (y_true * y_pred)))
    return (2. * intersection) / (sum(map(sum, y_true)) + sum(map(sum, y_pred)))


def get_img_resize(fn, image_type, DIM=(WIDTH, HEIGHT)):
    """
    function to get mask image and return as array
    """
    if "mask" in image_type:
        fname = get_filename(fn, "Train_mask")
    elif "Test" in image_type:
        fname = get_filename(fn, "Test")
    else:
        fname = get_filename(fn, "Train")
    img_pil = Image.open(fname)
    mask = img_pil.resize(DIM)
    return np.asarray(mask)


def get_prediction_dice(ids, prediction, DIM=(WIDTH, HEIGHT)):
    """
    calculates dice coefficient for each item and its predicted mask
    """
    results = []
    for fn, pred in zip(ids, prediction):
        # get masks corresponding to the fname (id)
        mask = get_img_resize(fn, "Train_mask", DIM)

        # get prediction as array
        pred_img = np.reshape(pred, DIM)
        pred_img = pred_img > 0.5

        results.append(dice(mask, pred_img))
    return results


def get_predicted_img(output, DIM=(WIDTH, HEIGHT)):
    data = np.reshape(output, DIM)
    mask = data > 0.5
    return np.asarray(Image.fromarray(mask, 'L'))


train_dice = get_prediction_dice(train_ids, predict)

idxmin_train = np.argmin(train_dice)

print("Smallest dice coefficient found in training set is {:.4f}. The file name is {}.jpg".format(
    train_dice[idxmin_train], train_ids[idxmin_train]))

# Show the mask that has the smallest dice coefficient
# plt.imshow(get_predicted_img(predict[idxmin_train]))

# The original image
# img = get_img_resize(train_ids[idxmin_train], "Train")
# plt.imshow(img)


def show_pic_and_original_mask(image_id):
    """
    Helper function to plot original image, original mask and bitwise picture
    """
    # load data
    img = get_img_resize(image_id, "Train")
    mask = get_img_resize(image_id, "Train_mask")
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    print("Image shape: {} | image type: {} | mask shape: {} | mask type: {}".format(img.shape, img.dtype, mask.shape,
                                                                                     mask.dtype))

    # plot left pic
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Original")

    # plot middle pic
    plt.subplot(132)
    plt.imshow(mask)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("True Mask")

    # plot right pic
    plt.subplot(133)
    plt.imshow(img_masked)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Original + Original Mask")


def show_pic_and_predicted_mask(image_id, pred_mask, dicecoef):
    # load data
    img = get_img_resize(image_id, "Train")
    mask = get_predicted_img(pred_mask)
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    print("Dice Coefficient: {:.4f}".format(dicecoef))

    # plot left pic
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Original")

    # plot middle pic
    plt.subplot(132)
    plt.imshow(mask)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Predicted Mask")

    # plot right pic
    plt.subplot(133)
    plt.imshow(img_masked)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Original + Pred Mask")


sorted_train_dice, sorted_train_id, sorted_predict = zip(*sorted(zip(train_dice, train_ids, predict)))

# todo:
show_pic_and_original_mask(train_ids[idxmin_train])
show_pic_and_predicted_mask(sorted_train_id[0], sorted_predict[0], sorted_train_dice[0])


def show_diff(image_id, pred_mask):
    """
    Show the difference of the predicted mask and true (original mask)
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(abs(get_predicted_img(pred_mask) - get_img_resize(image_id, "Train_mask")))
    plt.xticks([])
    plt.yticks([])


show_diff(sorted_train_id[0], sorted_predict[0])

# Missing window details and antenna

# todo:
show_pic_and_original_mask(sorted_train_id[1])
show_pic_and_predicted_mask(sorted_train_id[1], sorted_predict[1], sorted_train_dice[1])

show_diff(sorted_train_id[1], sorted_predict[1])

# Some minor outline was not predicted

# Get prediction - validation set

val_data_gen = test_datagen.flow_from_directory(
    validate_dir,
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    shuffle=False,
    class_mode=None,
    batch_size=BATCH_SIZE)

filenames = val_data_gen.filenames
nb_samples = len(filenames)

val_ids = [fn.split('\\')[-1][:-4] for fn in filenames]

predict_val = model.predict(val_data_gen, steps=
np.ceil(nb_samples / BATCH_SIZE))

with open(main_dir + '/output/val-prediction.txt', 'wb') as file:
    file.write(predict_val)

with open(main_dir + '/output/val-prediction_name.txt', 'w') as file:
    for line in val_ids:
        file.write(line + '\n')

predict_dice = get_prediction_dice(val_ids, predict_val)
sorted_val_dice, sorted_val_id, sorted_val_predict = zip(*sorted(zip(predict_dice, val_ids, predict_val)))

# Show 5 pictures with the lowest dice coefficient
for i in range(5):
    show_pic_and_original_mask(sorted_val_id[i])
    # show_pic_and_predicted_mask(sorted_val_id[i], sorted_val_predict[i], sorted_val_dice[i])
    show_diff(sorted_val_id[i], sorted_val_predict[i])


# Common prediction mistake:
#
# * Missing Antenna (too fine a detail)
# * Missing detail when car color is close to environment color (i.e. black details near shadow or white details near white background)
# * Borderlines

# Organize the pictures slightly better

def show_pic_and_original_mask_and_predicted(image_id, pred_mask, dicecoef):
    """
    Helper function to plot original image, original mask and predicted mask
    """
    # load data
    img = get_img_resize(image_id, "Train")
    mask = get_img_resize(image_id, "Train_mask")
    pmask = get_predicted_img(pred_mask)
    diff = Image.fromarray(abs(pmask - mask), 'L')

    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.imshow(img)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Dice Coef: {:.4f}".format(dicecoef))

    plt.subplot(142)
    plt.imshow(mask)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("True Mask")

    plt.subplot(143)
    plt.imshow(pmask)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Predicted Mask")

    plt.subplot(144)
    plt.imshow(diff)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Highlighted difference")


# Train set high error pics
# TODO:
show_pic_and_original_mask_and_predicted(sorted_train_id[0], sorted_predict[0], sorted_train_dice[0])
show_pic_and_original_mask_and_predicted(sorted_train_id[1], sorted_predict[1], sorted_train_dice[1])
show_pic_and_original_mask_and_predicted(sorted_train_id[2], sorted_predict[2], sorted_train_dice[2])
# Now validation set
for i in range(5):
    show_pic_and_original_mask_and_predicted(sorted_val_id[i], sorted_val_predict[i], sorted_val_dice[i])

# Predict for unseen data
#
# Try to predict a car that the data has never seen before; We chose 18 pictures from the original test set (the original test set is too big and would take hours to predict it one by one)

# todo:
# test_dir = DATA_PATH + '/test/'
test_dir = DATA_PATH + '/val_images/'

test_data_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    shuffle=False,
    class_mode=None,
    batch_size=BATCH_SIZE)

filenames = test_data_gen.filenames
nb_samples = len(filenames)

test_ids = [fn.split('\\')[-1][:-4] for fn in filenames]

test_predict = model.predict(test_data_gen, steps=np.ceil(nb_samples / BATCH_SIZE))

with open(main_dir + '/output/test-prediction.txt', 'wb') as file:
    file.write(test_predict)

with open(main_dir + '/output/test-prediction_name.txt', 'w') as file:
    for line in test_ids:
        file.write(line + '\n')


def show_test_pic_and_predicted_mask(image_id, pred_mask):
    # load data
    img = get_img_resize(image_id, "Test")
    mask = get_predicted_img(pred_mask)
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    # plot left pic
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Original")

    # plot middle pic
    plt.subplot(132)
    plt.imshow(mask)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Predicted Mask")

    # plot right pic
    plt.subplot(133)
    plt.imshow(img_masked)
    # remove ticks for pictures
    plt.xticks([])
    plt.yticks([])
    plt.title("Original + Pred Mask")


# TODO:
for i in range(nb_samples):
    show_test_pic_and_predicted_mask(test_ids[i], test_predict[i])

# The outcome looks decent despite some details still missing (dark color near shadow area, fine details etc.)

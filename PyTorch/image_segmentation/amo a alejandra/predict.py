# USAGE
# python predict.py
from bajista import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


def prepare_plot(orig_image, orig_mask, pred_mask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(orig_image)
    ax[1].imshow(orig_mask)
    ax[2].imshow(pred_mask)

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()
    plt.show()


def make_predictions(model, image_path):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()

        # find the filename and generate the path to ground truth mask
        filename = image_path.split(os.path.sep)[-1]
        ground_truth_path = os.path.join(config.MASK_DATASET_PATH, filename)

        # load the ground-truth segmentation mask in grayscale mode and resize it
        gt_mask = cv2.imread(ground_truth_path, 0)

        # Says height and height, but it does work.
        gt_mask = cv2.resize(gt_mask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        pred_mask = model(image).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()

        # filter out the weak predictions and convert them to integers
        pred_mask = (pred_mask > config.THRESHOLD) * 255
        pred_mask = pred_mask.astype(np.uint8)

        # prepare a plot for visualization
        prepare_plot(orig, gt_mask, pred_mask)


# load the image paths in our testing file and randomly select 10 image paths
print("\n[INFO] loading up test image paths...")
image_paths = open(config.TEST_PATHS).read().strip().split("\n")
image_paths = np.random.choice(image_paths, size=10)
# image_paths = ["data/train/images/7cc53fe88b.png", "data/train/images/9ca520f895.png"]

# load our model from disk and flash it to the current device
print("\n[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
# print("unet", unet)

# iterate over the randomly selected test image paths
for path in image_paths:
    # make predictions and visualize the results
    make_predictions(unet, path)

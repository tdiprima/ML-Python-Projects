# USAGE
# python train.py
import os
import sys
import time  # timing our training process

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from imutils import paths  # convenience functions to make basic image processing functions
from sklearn.model_selection import train_test_split  # split our dataset into training and testing sets
from torch.nn import BCEWithLogitsLoss  # using binary cross-entropy loss to train our model
from torch.optim import Adam  # to train our network
from torch.utils.data import DataLoader
from torchvision import transforms  # apply image transformations on our input images
from tqdm import tqdm  # keeping track of progress during training

from pyimagesearch import config
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet

# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths,
                         test_size=config.TEST_SPLIT, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

# TODO: this seems really dumb...
# write the testing image paths to disk so that we can use then when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

# Define the transformations that we want to apply while loading our input images and consolidate them.
# We used OpenCV to load images in our custom dataset, but PyTorch expects the input image samples to be in PIL format.
transforms = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                                                    config.INPUT_IMAGE_WIDTH)),
                                 transforms.ToTensor()])
# ToTensor(): enables us to convert input images to PyTorch tensors and convert the input PIL Image,
# which is originally in the range from [0, 255], to [0, 1].

# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
                              transforms=transforms)

testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
                             transforms=transforms)

print(f"\n[INFO] found {len(trainDS)} examples in the training set...")
print(f"\n[INFO] found {len(testDS)} examples in the test set...")

# shuffle = True in the train dataloader since we want samples from all classes to be
# uniformly present in a batch, which is important for optimal learning and convergence
# of batch gradient-based optimization approaches.
trainLoader = DataLoader(trainDS, shuffle=True,
                         batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=os.cpu_count())

# create our test dataloader
testLoader = DataLoader(testDS, shuffle=False,
                        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                        num_workers=os.cpu_count())

# Initialize our U-Net model and the training parameters.
unet = UNet().to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()

# The Adam optimizer class takes the parameters of our model and the learning rate
# that we will be using to train our model.
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# Calculate the number of steps required to iterate over our entire train and test set.
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

# Create an empty dictionary to keep track of our training and test loss history.
H = {"train_loss": [], "test_loss": []}

# TRAINING LOOP

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()

try:
    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        unet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far
            totalTrainLoss += loss

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()

            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        # print the model training and validation information
        print("\n[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("\nType", exc_type)
    print("\nErr:", exc_obj)
    print("\nLine:", exc_tb.tb_lineno)
    sys.exit(1)

# display the total time needed to perform the training
endTime = time.time()

print("\n[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
plt.show()  # TODO: why they not put this?

# serialize the model to disk
torch.save(unet, config.MODEL_PATH)


# USE TRAINED MODEL FOR PREDICTION

def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)

    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()

        # find the filename and generate the path to ground truth mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
                                       filename)

        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
                                     config.INPUT_IMAGE_HEIGHT))

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

        # filter out the weak predictions and convert them to integers
        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)

        # prepare a plot for visualization
        prepare_plot(orig, gtMask, predMask)


# load the image paths in our testing file and randomly select 10
# image paths
print("\n[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

# iterate over the randomly selected test image paths
for path in imagePaths:
    # make predictions and visualize the results
    make_predictions(unet, path)

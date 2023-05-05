# USAGE
# python train.py
# https://www.kaggle.com/code/tammydiprima/pytorch-unet/
from tools.dataset import SegmentationDataset
from tools.model import UNet
from tools import config
from torch.nn import BCEWithLogitsLoss  # using binary cross-entropy loss to train our model
from torch.optim import Adam  # to train our network
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split  # split our dataset into training and testing sets
from torchvision import transforms  # apply image transformations on our input images
from imutils import paths  # convenience functions to make basic image processing functions
from tqdm import tqdm  # keeping track of progress during training
import matplotlib.pyplot as plt
import torch
import time  # timing our training process

NUM_WORKERS = 0

# load the image and mask filepaths in a sorted manner
image_paths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
mask_paths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

if len(image_paths) == 0:
    print("\nGot Data?\n")
    exit(1)

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(image_paths, mask_paths,
                         test_size=config.TEST_SPLIT, random_state=42)

# unpack the data split
(train_images, testImages) = split[:2]
(train_masks, testMasks) = split[2:]

# write the testing image paths to disk so that we can use them when evaluating/testing our model
print("\n[INFO] saving testing image paths...")
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
train_ds = SegmentationDataset(image_paths=train_images, mask_paths=train_masks,
                              transforms=transforms)

test_ds = SegmentationDataset(image_paths=testImages, mask_paths=testMasks,
                             transforms=transforms)

print(f"\n[INFO] found {len(train_ds)} examples in the training set...")
print(f"[INFO] found {len(test_ds)} examples in the test set...")

# shuffle = True in the train dataloader since we want samples from all classes to be
# uniformly present in a batch, which is important for optimal learning and convergence
# of batch gradient-based optimization approaches.
train_loader = DataLoader(train_ds, shuffle=True,
                         batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                         num_workers=NUM_WORKERS)

# create our test dataloader
test_loader = DataLoader(test_ds, shuffle=False,
                        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                        num_workers=NUM_WORKERS)

# Initialize our U-Net model and the training parameters.
unet = UNet().to(config.DEVICE)

# initialize loss function and optimizer
loss_func = BCEWithLogitsLoss()

# The Adam optimizer class takes the parameters of our model and the learning rate
# that we will be using to train our model.
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# Calculate the number of steps required to iterate over our entire train and test set.
train_steps = len(train_ds) // config.BATCH_SIZE
test_steps = len(test_ds) // config.BATCH_SIZE

# Create an empty dictionary to keep track of our training and test loss history.
H = {"train_loss": [], "test_loss": []}

# TRAINING LOOP

# loop over epochs
print("[INFO] training the network...")
start_time = time.time()

for e in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    unet.train()

    # initialize the total training and validation loss
    total_train_loss = 0
    total_test_loss = 0

    # loop over the training set
    for (i, (x, y)) in enumerate(train_loader):
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        # perform a forward pass and calculate the training loss
        pred = unet(x)
        loss = loss_func(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

        # add the loss to the total training loss so far
        total_train_loss += loss

    # switch off autograd
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()

        # loop over the validation set
        for (x, y) in test_loader:
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # make the predictions and calculate the validation loss
            pred = unet(x)
            total_test_loss += loss_func(pred, y)

    # calculate the average training and validation loss
    avg_train_loss = total_train_loss / train_steps
    avg_test_loss = total_test_loss / test_steps

    # update our training history
    H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
    H["test_loss"].append(avg_test_loss.cpu().detach().numpy())

    # print the model training and validation information
    print("\n[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(avg_train_loss, avg_test_loss))

# display the total time needed to perform the training
end_time = time.time()
print("\n[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

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
plt.show()

# serialize the model to disk
torch.save(unet, config.MODEL_PATH)

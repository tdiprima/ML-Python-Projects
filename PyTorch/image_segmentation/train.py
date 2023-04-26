"""
Main code; start here.
https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# The model module is a custom module defined model.py
from model import UNET

# Custom module from utils.py
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# TODO:
# NUM_EPOCHS = 3
NUM_EPOCHS = 1

# TODO:
# NUM_WORKERS = 2
NUM_WORKERS = 4

# You'd want to change it to original size, and then resize to nearest interpolation.
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally

PIN_MEMORY = True

# TODO:
# LOAD_MODEL = False
LOAD_MODEL = True

# TODO:
TRAIN_IMG_DIR = "../data/train_images/"
TRAIN_MASK_DIR = "../data/train_masks/"
VAL_IMG_DIR = "../data/val_images/"
VAL_MASK_DIR = "../data/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    For training (1) epoch.
    """
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # float for binary cross-entropy loss
        tensor_a = targets.float()
        print("targets.float", tensor_a.shape)

        # un-squeeze 1 (adding a channel dimension)
        tensor_b = tensor_a.unsqueeze(1)
        print("targets un-squeeze", tensor_b.shape)

        targets = tensor_b.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            """
            Use the computer's GPU to do math quickly and accurately.
            Use float-16 training, reducing VRAM (video RAM) & speeding up training.
            """
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()  # Zero all the gradients from previous.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    print("=> Creating train and val transforms")
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # If you wanted multiclass segmentation, you could do:
    # out_channels=3 (eg. 3 colors, 2 classes), and loss fn to cross entropy loss
    print("=> Creating model")
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    # You could use BCELoss() if you did: return torch.sigmoid(self.final_conv(x)) on the output of the model.
    loss_fn = nn.BCEWithLogitsLoss()  # bc we're not doing sigmoid on our output

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    print("=> Starting training")
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy (validation loader)
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

    import sys
    try:
        from torchsummary import summary
        # (channels, height, width)
        summary(model, (3, IMAGE_HEIGHT, IMAGE_WIDTH))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("\nType", exc_type)
        print("\nErr:", exc_obj)
        print("\nLine:", exc_tb.tb_lineno)
        sys.exit(1)


if __name__ == "__main__":
    # So that you don't get any issues when running NUM_WORKERS.
    main()

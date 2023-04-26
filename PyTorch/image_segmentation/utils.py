import torch
import torchvision
from torch.utils.data import DataLoader

# The dataset module is a custom module defined in dataset.py
from dataset import CarvanaDataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    print("=> Getting loaders")
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    """
    For segmentations, we're outputting a prediction for each individual pixel.
    And the class for each individual pixel.
    Adapt function if you have more classes.
    This is for binary.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        """
        Predictions [with torch.inference_mode()]
        """
        for x, y in loader:
            x = x.to(device)
            # Un-squeeze bc the label doesn't have a channel (bc it's grayscale)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))

            # All those that are higher than 0.5, convert to 1.
            preds = (preds > 0.5).float()
            # All less, convert to zero. todo: where is it?

            num_correct += (preds == y).sum()  # sum all the correct pixels
            num_pixels += torch.numel(preds)  # number of elements

            # Dice is "better metrics"; this one's for binary (vs multi classes)
            # We're summing the # of pix where they're both the same, or both outputting a white pixel (1).
            # Divide by the # of pix where they're outputting 1 for both of them.
            # Add an epsilon of 1e-8.
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    # HEADS UP.  Output just black pixels, accuracy > 80%
    # So there are better metrics for measuring this.  Obj detection, Intersection / Union, etc.
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()  # Set model.train() again.


def save_predictions_as_imgs(
        loader, model, folder="saved_images", device="cuda"
):
    """
    Get a visualization of what it's doing.
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Output prediction + corresponding correct one
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()

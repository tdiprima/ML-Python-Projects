import sys

import torch
import torchvision

assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# Continue with regular imports
import torch

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except Exception as e:
    print("[INFO] Couldn't find torchinfo... installing it.")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(exc_type, exc_obj, exc_tb.tb_lineno)
    sys.exit(1)

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular import data_setup, engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

from pathlib import Path

# Setup data path
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"  # images from a subset of classes from the Food101 dataset

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists, skipping re-download.")
else:
    print(f"Did not find {image_path}, downloading it...")

# Setup directory path
train_dir = image_path / "train"
test_dir = image_path / "test"

from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # resize image to 224, 224 (height x width)
    transforms.ToTensor(),  # get images into range [0, 1]
    normalize])  # make sure images have the same distribution as ImageNet

from going_modular import data_setup

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=manual_transforms,
                                                                               batch_size=32)
print("Success.")

# todo
# Get a set of pretrained model weights
# weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT  # "DEFAULT" = best available weights

# Get the transforms used to create our pretrained weights
# auto_transforms = weights.transforms()

# Create DataLoaders using automatic transforms
# train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
#                                                                                test_dir=test_dir,
#                                                                                transform=auto_transforms,
#                                                                                batch_size=32)
# /todo

# New method of creating a pretrained model (torchvision v0.13+)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT  # ".DEFAULT" = best available weights
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# print(f"\nmodel:", model)
print(f"\nclassifier:", model.classifier)

# Print with torchinfo
# from torchinfo import summary

# summary(model=model,
#         input_size=(1, 3, 224, 224),  # example of [batch_size, color_channels, height, width]
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# With a feature extractor model, typically you will "freeze" the base layers of a pretrained/foundation model and
# update the output layers to suit your own problem.

# Freeze all of the base layers in EffNetB0
for param in model.features.parameters():
    # print(param)
    param.requires_grad = False

print(f"\nGweat.", len(class_names))

# Update the classifier head of our model to suit our problem
from torch import nn

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,  # feature vector coming in
              out_features=len(class_names))).to(device)  # how many classes do we have?

print(f"\nuh-oh?", model.classifier)

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Import train function
from going_modular import engine

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer

start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

# Evaluate model by plotting loss curves
sys.path.append('../toolbox')
from helper_functions import plot_loss_curves

# Plot the loss curves of our model
plot_loss_curves(results)

# ADDED image_size
from matplotlib import pyplot as plt

from typing import List, Tuple

from PIL import Image

from torchvision import transforms


# 1. Take in a trained model...
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = device):
    # 2. Open the image with PIL
    img = Image.open(image_path)

    # 3. Create a transform if one doesn't exist
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # PREDICT ON IMAGE
    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on inference mode and eval mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform the image and add an extra batch dimension
        transformed_image = image_transform(img).unsqueeze(dim=0)  # [batch_size, color_channels, height, width]

        # 7. Make a prediction on the transformed image by passing it to the model (also ensure it's on the target device)
        target_image_pred = model(transformed_image.to(device))

        # 8. Convert the model's output logits to pred probs
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    # print("\npred max", target_image_pred_probs.max())

    # 9. Convert the model's pred probs to pred labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
    plt.show()


# Get a random list of image paths from the test set
import random

num_images_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_image_path_sample = random.sample(population=test_image_path_list,
                                       k=num_images_to_plot)

# Make predictions on and plot the images
for image_path in test_image_path_sample:
    pred_and_plot_image(model=model,
                        image_path=image_path,
                        class_names=class_names,
                        image_size=(224, 224))

# Making predictions on a custom image
# Setup custom image path
custom_image_path = data_path / "modified-pbbffft.jpeg"
print("\ncustom_image_path:", custom_image_path)

# Predict on custom image
pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names)

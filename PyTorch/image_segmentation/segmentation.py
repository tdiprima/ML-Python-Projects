"""
Loads a dataset of images and their associated labels, and trains a Fully Convolutional Network
(FCN) model for slide image segmentation.
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


class SlideImageDataset(Dataset):
    def __init__(self, image_folder_path, label_folder_path, transform=None):
        self.image_folder_path = image_folder_path
        self.label_folder_path = label_folder_path
        self.image_filenames = os.listdir(image_folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder_path, self.image_filenames[idx])
        label_path = os.path.join(self.label_folder_path, self.image_filenames[idx].split('.')[0] + '_label.png')

        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


class SlideImageSegmentationModel(nn.Module):
    def __init__(self):
        """
        TO BE U-NET
        """
        super(SlideImageSegmentationModel, self).__init__()
        # TODO: Define your model architecture here

    def forward(self, x):
        # TODO: Define the forward pass of your model here
        return x


class FCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        HOW ABOUT A FCN?
        PyTorch FCN Image Segmentation
        """
        super(FCN, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn_up3 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn_up2 = nn.BatchNorm2d(64)
        self.upconv1 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)

        # Decoder
        x = self.upconv3(x)
        x = self.bn_up3(x)
        x = nn.functional.relu(x)
        x = self.upconv2(x)
        x = self.bn_up2(x)
        x = nn.functional.relu(x)
        x = self.upconv1(x)

        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Define hyperparameters
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# Create instances of the dataset
train_dataset = SlideImageDataset('path/to/image/folder', 'path/to/label/folder', transform=transforms.ToTensor())

# Create instances of the model, loss function, and optimizer
model = SlideImageSegmentationModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create instances of the dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_model(model, train_loader, criterion, optimizer, num_epochs)

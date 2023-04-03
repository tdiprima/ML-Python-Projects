"""
tensorboard --logdir=runs
http://localhost:6006/
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md
"""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST


class ConvNet(nn.Module):
    """
    Simple Convolutional Neural Network for classifying MNIST digits
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 5, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 24 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)


def weight_histograms_conv2d(writer, step, weights, layer_number):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"layer_{layer_number}/kernel_{k}"
        writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
    flattened_weights = weights.flatten()
    tag = f"layer_{layer_number}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model):
    print("Visualizing model weights...")
    # Iterate over all model layers
    for layer_number in range(len(model.layers)):
        # Get layer
        layer = model.layers[layer_number]
        # Compute weight histograms for appropriate layer
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight
            weight_histograms_conv2d(writer, step, weights, layer_number)
        elif isinstance(layer, nn.Linear):
            weights = layer.weight
            weight_histograms_linear(writer, step, weights, layer_number)


if __name__ == '__main__':
    # Initialize the SummaryWriter for TensorBoard
    # Its output will be written to ./runs/
    writer = SummaryWriter()

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare CIFAR-10 dataset
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    # Initialize the ConvNet
    convnet = ConvNet()

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(convnet.parameters(), lr=1e-4)

    # Run the training loop
    loss_idx_value = 0
    for epoch in range(0, 5):  # 5 epochs at maximum
        loss_idx_value = 0  # todo: But...

        # Visualize weight histograms
        weight_histograms(writer, epoch, convnet)

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            if i > 1000:
                break

            # Get inputs
            inputs, targets = data

            # Write the network graph at epoch 0, batch 0
            if epoch == 0 and i == 0:
                writer.add_graph(convnet, input_to_model=data[0], verbose=True)

            # Write an image at every batch 0
            if i == 0:
                writer.add_image("Example input", inputs[0], global_step=epoch)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = convnet(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            writer.add_scalar("Loss", current_loss, loss_idx_value)
            loss_idx_value += 1
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

        # Write loss for epoch
        writer.add_scalar("Loss/Epochs", current_loss, epoch)

    # Process is complete.
    print('Training process has finished.')

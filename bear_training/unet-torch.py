"""
Basic U-Net architecture
Takes a 3-channel input image and outputs a single-channel segmentation mask.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    # The DoubleConv class defines a module that applies two convolutional layers with batch normalization and ReLU activation.
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    # The UNet class defines the U-Net architecture, with the features parameter specifying the number of channels for each layer.
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()  # contains the up-sampling blocks.
        self.downs = nn.ModuleList()  # contains the down-sampling blocks.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down-sampling part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up-sampling part of U-Net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # The bottleneck block is the central block in the network that connects the downsampling and upsampling paths
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Perform the final convolution to produce the segmentation mask
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # The forward method performs the forward pass of the U-Net network
    def forward(self, x):
        # The skip_connections list stores the outputs of the down-sampling blocks for later use in the up-sampling path.
        skip_connections = []

        # Down part of U-Net
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck of U-Net (connects the down-sampling and up-sampling paths)
        x = self.bottleneck(x)

        # Up part of U-Net
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[len(self.ups) // 2 - idx // 2 - 1]
            x = torch.cat([skip_connection, x], dim=1)
            x = self.ups[idx + 1](x)

        # Final convolution
        x = self.final_conv(x)
        return x


import matplotlib.pyplot as plt
import sys


# Display results
def display_results(model, test_image):
    try:
        # Make a prediction on the test image
        output = model(test_image)
        prediction = output[-1].item()

        # Plot the original image, ground truth, and predicted mask
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(test_image[0, :, :, 0], cmap='gray')
        ax[0].set_title('Input Image')
        ax[1].imshow(test_mask[0, :, :, 0], cmap='gray')
        ax[1].set_title('Ground Truth')
        ax[2].imshow(prediction[0, :, :, 0], cmap='gray')
        ax[2].set_title('Prediction')
        plt.show()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("\nType", exc_type)
        print("\nErr:", exc_obj)
        print("\nLine:", exc_tb.tb_lineno)
        sys.exit(1)


# Display results
def display_results1(model, test_image):
    output = model(test_image)
    # Get the prediction tensor, detach it from the computation graph, move to cpu, convert to numpy array, and squeeze it
    prediction = output.detach().cpu().numpy().squeeze()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(test_image.cpu().numpy()[0].transpose((1, 2, 0)))  # converting tensor to numpy and rearranging dims for display
    ax[0].set_title('Input Image')

    ax[2].imshow(prediction, cmap='gray')
    ax[2].set_title('Prediction')
    plt.show()


def display_results2(model, test_image, test_mask):
    output = model(test_image)
    prediction = output.detach().cpu().numpy()
    prediction = prediction.reshape(prediction.shape[2], prediction.shape[3])  # reshape the output

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(test_image[0, 0, :, :], cmap='gray')
    ax[0].set_title('Input Image')
    ax[1].imshow(test_mask[0, 0, :, :], cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[2].imshow(prediction, cmap='gray')
    ax[2].set_title('Prediction')
    plt.show()


model = UNet()  # Using the defaults


def dummy1():
    # Test the model with dummy image data
    test_image = torch.ones((256, 256, 3))  # shape: [256, 256, 3]
    test_mask = torch.zeros((256, 256, 3))  # shape: [256, 256, 3]

    # Adding the batch dimension and rearranging the dimensions to (N, C, H, W)
    test_image = test_image.unsqueeze(0).permute(0, 3, 1, 2)
    test_mask = test_mask.unsqueeze(0).permute(0, 3, 1, 2)
    # print(test_image.shape)  # [1, 3, 256, 256]

    display_results1(model, test_image)


def dummy2():
    # Test the model with a sample image
    from PIL import Image
    from tensorflow.keras.preprocessing import image

    test_image = Image.open("Formula1.jpg")
    img = image.img_to_array(test_image)
    img = img.reshape((1,) + img.shape)
    test_image = img

    test_image = torch.from_numpy(test_image)  # Shape: [1, 256, 256, 3]
    test_image = test_image.permute(0, 3, 1, 2)  # Shape: [1, 3, 256, 256]

    # test_mask = test_image.clone()
    display_results1(model, test_image)


def dummy3():
    # And again test the model with a sample image
    from PIL import Image
    from tensorflow.keras.preprocessing import image

    test_image = Image.open("Formula1.jpg")
    img = image.img_to_array(test_image)
    img = img.reshape((1,) + img.shape)
    tensor = torch.from_numpy(img).float()

    # permute the tensor to match PyTorch's expected input shape
    test_image = tensor.permute(0, 3, 1, 2)

    # I'm using test_image as test_mask for demonstration, replace it with your actual mask
    test_mask = test_image.clone()

    display_results2(model, test_image, test_mask)


if __name__ == "__main__":
    # dummy1()
    dummy2()
    # dummy3()

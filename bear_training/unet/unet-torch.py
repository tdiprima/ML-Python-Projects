"""
Basic U-Net architecture
Takes a 3-channel input image and outputs a single-channel segmentation mask.
Unet for Image Analysis.md
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    The DoubleConv class defines a module that applies two convolutional layers
    with batch normalization and ReLU activation.
    """
    print("DoubleConv")

    def __init__(self, in_channels, out_channels):
        print("__init__")
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
        print("forward")
        return self.conv(x)


class UNet(nn.Module):
    """
    The UNet class defines the U-Net architecture, with the features parameter
    specifying the number of channels for each layer. (Orly?)
    """
    print("UNet")

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        print("init 1")
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
        print("forward 1")
        """
        The skip_connections list stores the outputs of the down-sampling blocks 
        for later use in the up-sampling path.
        """
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

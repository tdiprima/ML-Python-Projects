"""
Defines a U-Net architecture for binary image segmentation and includes a test function to verify its functionality.
The U-Net is composed of an encoder (downsampling path) and decoder (upsampling path), with a bottleneck layer in between.
The architecture also includes skip connections for preserving detailed input information.
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # kernel size: 3, stride: 1, padding: 1
        # bias: False, because it'll be cancelled by BatchNorm
        # same convolution: img h / w be the same after the convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    """
    Binary image segmentation; out_channels: 1.
    """
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()  # Instead of a regular list.
        self.downs = nn.ModuleList()

        # Common: down-sampling using a convolutional layer with a stride greater than 1
        # Reduces the spatial size of the input feature map by a factor of 2:
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # eg. 161 x 161, output: 160 x 160 (MaxPool floors the shapes)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))  # The DoubleConv up there.
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            # ConvTranspose2d + DoubleConv
            # To up-sample the feature maps, one can use a transposed convolutional layer.
            # Doubles the spatial size of the input feature map:
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            # "append", bc we're gonna do up, and 2 Convs.
            self.ups.append(DoubleConv(feature * 2, feature))

        # features[-1] bc we want this 512 (the last in our features list)
        # and map it to features[-1] * 2 (1024)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # Final: 1 x 1 conv; doesn't change img h/w.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)  # Add skip conn b4 down sampling.
            # First with highest res, last with smallest res.
            x = self.pool(x)

        x = self.bottleneck(x)

        """
        We wanna go backwards in that order when doing the concatenation;
        first element = highest res, so reverse that list with [::-1].
        """
        skip_connections = skip_connections[::-1]

        # Step of 2 (bc wanna do the up, then double-conv = single step)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # Index integer division by 2, bc step 2.
            skip_connection = skip_connections[idx // 2]

            # What if input isn't perfectly divisible by 2...
            if x.shape != skip_connection.shape:
                # Take out h/w, skipping batch size and num channels [2:]
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Add along channel dimension (batch, channel, height, width)
            # Concatenate the feature maps from an earlier layer with those from a later layer,
            # in order to preserve more detailed information from the input image.
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # Get skip, concatenate, run it through double-conv.
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    """
    Batch size: 3, input: 1 channel, size
    """
    # Create dummy tensor with shape
    # x = torch.randn((3, 1, 160, 160))  # perfectly divisible
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print("\npreds shape:", preds.shape)
    print("input shape:", x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()

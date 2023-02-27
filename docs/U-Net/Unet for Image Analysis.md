## U-Net

<span style="color: #0000dd; font-size: larger">U-Net is a convolutional neural network that was developed for biomedical image segmentation<!-- at the Computer Science Department of the University of Freiburg-->.</span>

Imagine you're drawing a picture, but you don't want to color the entire thing. Instead, you want to color just one part of it, like a flower or a tree.

Now, let's say you have a friend who's really good at coloring. You want your friend to color that one part of the picture for you, but you need to tell your friend exactly where to color.

This is kind of like what U-Net does in machine learning! U-Net is a type of neural network that helps computers "color" specific parts of an image.

It's often used in **medical imaging** to **highlight certain parts** of an X-ray or MRI scan.

### U-Net works by...

* Taking an image and breaking it down into smaller pieces, sort of like a puzzle. <span style="font-size: 27px;">🧩</span>
* Then, it analyzes each piece to figure out which parts of the image are important and should be highlighted. <span style="font-size: 27px;">🔍</span>
* Finally, it puts all the pieces back together and shows you the final result. <span style="font-size: 27px;">🏞️</span>

So, just like how you need to tell your friend exactly where to color in your picture, U-Net uses a lot of complicated math to figure out which parts of an image are important to highlight. 

But once it's done, it can help doctors and researchers better understand what's going on inside the human body!

## U-Net Implementation

Here is a basic example of implementing a U-Net architecture using PyTorch:

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
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
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down part of U-Net
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck of U-Net
        x = self.bottleneck(x)

        # Up part of U-Net
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[len(self.ups)//2 - idx//2 - 1]
            x = torch.cat([skip_connection, x], dim=1)
            x = self.ups[idx+1](x)

        # Final convolution
        x = self.final_conv(x)
        return x
```

<br>

This code defines a basic U-Net architecture that takes a **3-channel input image** and outputs a **single-channel segmentation mask.**

It consists of a downsampling path and an upsampling path, with skip connections between them.

<mark>The `DoubleConv` class defines a module that **applies two convolutional layers** with **batch normalization** and **ReLU activation.**</mark>

The `UNet` class defines the U-Net architecture, with the `features` parameter specifying the **number of channels** for each layer.

The `downs` list contains the **downsampling** blocks.

The `ups` list contains the **upsampling** blocks.

The `bottleneck` block is the central block in the network that **connects** the downsampling and upsampling paths.

The `forward` method performs the **forward pass** of the U-Net network.

The `skip_connections` list **stores the outputs** of the downsampling blocks for later use in the upsampling path.

The **bottleneck** block connects the downsampling and upsampling paths.

The **upsampling** blocks upsample the **feature maps** and **concatenate** them with the corresponding **skip connection** from the downsampling path.

Finally, the **`final_conv`** layer performs the final **convolution** to produce the **segmentation mask.**

<br>

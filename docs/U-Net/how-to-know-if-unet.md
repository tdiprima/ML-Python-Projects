## How do you know if it's U-Net?

* Down-sampling layer
* Up-sampling layer
* Skip connections
* Symmetric architecture


## PyTorch CNN Downsample

```py
import torch.nn as nn

downsample = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
```

<br>
This code defines a **2D convolutional layer**, often used in deep learning, particularly for **image-related** tasks.

This 2D convolutional layer, `downsample`, is named so because it effectively **reduces** the spatial resolution **(height and width)** of its input by a **factor of 2**. This is achieved by specifying a stride of 2, which means the convolutional filters move two pixels at a time instead of the default one pixel.

### Parameters

- `in_channels=64`: The number of input channels (**depth**) that the convolution layer accepts. In the context of an image, this could be 3 for an RGB image (red, green, blue) or 1 for a grayscale image.

    But in the deeper layers of a neural network, this could be more as each channel may represent some high-level feature. Here, it's set to 64, indicating that it expects 64 different feature maps.

- `out_channels=128`: The number of output channels (**filters**) that the convolution layer will learn. Each of these filters will be convolved with the input image (or feature map), producing 128 different feature maps as output.

- `kernel_size=3`: The **height and width** of the convolutional window (also known as a kernel or filter). In this case, it's a 3x3 window.

- `stride=2`: Reduces the spatial size of the input feature map by a factor of 2.  Stride 2 is the **step size** for moving the convolutional kernel across the input image or feature map.

    A stride of 2 means the window moves 2 pixels at a time. This is what makes the layer downsample *(verb)* its input, reducing the spatial dimensions (height and width) by approximately a **factor of 2**.

- `padding=1`: The number of zero-padding pixels added to **all sides** of the input. This is often used to preserve the spatial dimensions of the input, but in this case, it helps to make sure that **all pixels are visited** by the center of the convolutional kernel, while still downsampling the input due to the stride of 2.

So, in summary, the code defines a convolutional layer that will downsample its input by a factor of 2, while transforming 64-dimensional input feature maps to 128-dimensional output feature maps.

## Up-sampling layer

All of the above, put together.

To up-sample the feature maps, one can use a transposed convolutional layer, also known as a **deconvolutional** layer. 

Here's an example of a 2D transposed convolutional layer that **doubles** the spatial size of the input feature map:

```py
upsample = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
```

<br>
Same as before, except the input and output channel values are reversed now.  In 64, out 128... in 128, out 64.

## Skip connection

A skip connection is simply a way to concatenate the feature maps from an earlier layer with those from a later layer, in order to preserve more detailed information from the input image.

```py
x = downsample(input_tensor)
y = upsample(x)
# concatenate along the channel dimension
z = torch.cat([x, y], dim=1)
```

## Symmetric architecture

1. The encoder and decoder paths typically have the same number of layers

2. The number of feature maps is usually doubled or halved at each layer, respectively.

A symmetric **encoder-decoder** network:

```py
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Down-sampling layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # Up-sampling layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1)

        # Skip connections
        self.skip1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.skip2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.skip3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # It konked out at 'padding='.
        # Add 4th layer.
        self.skip4 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)


UNet(64, 128)
```

<br>

**Down-sampling layers** (such as convolutional layers, pooling layers, or strided convolutions) reduce the spatial size of the input image.

**Up-sampling layers** (such as transposed convolutional layers, upsampling layers, or bilinear interpolation layers) increase the spatial size of the feature maps.

**Skip connections** directly connect the output of a down-sampling layer to the corresponding up-sampling layer.

<br>

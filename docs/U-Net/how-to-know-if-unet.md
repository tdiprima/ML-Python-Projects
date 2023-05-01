### How do you know if a PyTorch program is using the UNet architecture?

Here are some examples of what each of these components might look like in PyTorch code:

## Down-sampling layer:

A common way to perform down-sampling in PyTorch is to use a <mark>**convolutional layer**</mark> with a stride greater than 1.

Here's an example of a 2D convolutional layer with a **stride of 2**, which reduces the spatial size of the input feature map by a factor of 2:

```ruby
import torch.nn as nn

downsample = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
```

<br>

## Up-sampling layer:

To up-sample the feature maps, one can use a <mark>**transposed**</mark> **convolutional layer**, also known as a deconvolutional layer. 

Here's an example of a 2D transposed convolutional layer that doubles the spatial size of the input feature map:

```ruby
upsample = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
```

<br>

## Skip connection:

A <mark>**skip connection**</mark> is simply a way to concatenate the feature maps from an earlier layer with those from a later layer, in order to preserve more detailed information from the input image.

```ruby
x = downsample(input_tensor)
y = upsample(x)
z = torch.cat([x, y], dim=1)  # concatenate along the channel dimension
```

<br>

## Symmetric architecture:

1. The encoder and decoder paths typically have the **same number of layers**

2. The number of feature maps is usually **doubled** or **halved** at each layer, respectively.

A symmetric encoder-decoder network:

```ruby
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
```

<br>

**Down-sampling layers** (such as convolutional layers, pooling layers, or strided convolutions) reduce the spatial size of the input image.

**Up-sampling layers** (such as transposed convolutional layers, upsampling layers, or bilinear interpolation layers) increase the spatial size of the feature maps.

**Skip connections** directly connect the output of a down-sampling layer to the corresponding up-sampling layer.

<br>

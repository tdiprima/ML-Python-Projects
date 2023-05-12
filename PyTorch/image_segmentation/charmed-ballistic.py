"""
Create random tensor and run it through unet model.
https://becominghuman.ai/implementing-unet-in-pytorch-8c7e05a121b4
https://github.com/spctr01/UNet/blob/master/Unet.py
"""
import torch
import torch.nn as nn
from torchinfo import summary


def dual_conv(in_channel, out_channel):
    """
    Double 3x3 convolution
    A Sequential layer of two convolution layers with kernel size 3 (3x3 conv) each followed by a relu activation
    Returns the "conv" a sequential layer
    """
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv


def crop_tensor(target_tensor, tensor):
    """
    Crop the image(tensor) to equal size
    Half left side image is concatenated with right side image
    """
    # Note! We are assuming height and width are the same size.
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    diff = tensor_size - delta

    # [:, :,] = all dimensions
    return tensor[:, :, delta:diff, delta:diff]


class Unet(nn.Module):
    """
    Original UNet Architecture
    """
    def __init__(self):
        super(Unet, self).__init__()
        """
        Left side (contracting path)
        Down conv (5 layers are on the left side)
        """
        self.dwn_conv1 = dual_conv(1, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        """
        Right side (expansion path)
        Up conv; transpose convolution
        """
        self.trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = dual_conv(1024, 512)
        self.trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = dual_conv(512, 256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = dual_conv(256, 128)
        self.trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = dual_conv(128, 64)

        # output layer
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, img):
        """
        We will forward pass the input(image) to the left side layers
        Forward pass for Left side
        """
        # print("\nInput size:", img.size())
        x1 = self.dwn_conv1(img)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)

        """
        Forward pass for Right side
        """
        x = self.trans1(x9)
        y = crop_tensor(x, x7)
        # Combine both the images using torch.cat() and pass it to up_conv()
        x = self.up_conv1(torch.cat([x, y], 1))

        x = self.trans2(x)
        y = crop_tensor(x, x5)
        x = self.up_conv2(torch.cat([x, y], 1))

        x = self.trans3(x)
        y = crop_tensor(x, x3)
        x = self.up_conv3(torch.cat([x, y], 1))

        x = self.trans4(x)
        y = crop_tensor(x, x1)
        x = self.up_conv4(torch.cat([x, y], 1))

        x = self.out(x)

        return x


if __name__ == '__main__':
    # batch, channels, rows, columns (height, width)
    image = torch.rand((1, 1, 572, 572))
    model = Unet()
    op_tensor = model(image)
    # print(f"\noutput shape: {op_tensor.shape}")  # torch.Size([1, 2, 388, 388])
    # print("\noutput:\n", op_tensor)
    print()
    summary(model, input_size=[1, 1, 572, 572])

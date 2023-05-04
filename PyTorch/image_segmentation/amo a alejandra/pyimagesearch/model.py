# Import the necessary layers, modules, and activation functions from PyTorch
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


# All models or model sub-parts are required to inherit from the PyTorch Module class
class Block(Module):
    """
    Take an input feature map with the inChannels number of channels,
    apply two convolution operations with a ReLU activation between them, and
    return the output feature map with the outChannels channels.
    """
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # Initialize the two convolution layers (self.conv1 and self.conv2)
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        # and a ReLU activation
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)

    def forward(self, x):
        # Take our feature map x, apply self.conv1 => self.relu => self.conv2
        # sequence of operations, and return the output feature map.
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        """
        Class constructor takes a tuple (channels) of channel dimensions
        Note that the first value denotes the number of channels in our input image,
        and the subsequent numbers gradually double the channel dimension.
        """
        super().__init__()

        # Each Block takes the input channels of the previous block and doubles
        # the channels in the output feature map.
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

        # Initialize a MaxPool2d() layer, which reduces the spatial dimension
        # of the feature maps by a factor of 2
        self.pool = MaxPool2d(2)

    def forward(self, x):
        """
        Initialize an empty list to store the intermediate outputs.
        Note that this will enable us to later pass these outputs to our
        decoder, where they can be processed with the decoder feature maps.
        """
        blockOutputs = []

        # loop through each block in our encoder
        for block in self.encBlocks:
            # Process the input feature map through the block
            x = block(x)
            # Add the output of the block to our blockOutputs list
            blockOutputs.append(x)
            # Apply the max pool operation on our block output
            x = self.pool(x)

        # return the list containing the intermediate outputs
        return blockOutputs


class Decoder(Module):
    # The channels gradually decrease by a factor of 2 instead of increasing.

    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels

        # Define a list of upsampling blocks that use the ConvTranspose2d layer
        # to upsample the spatial dimension of the feature maps by a factor of 2.
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)]
        )

        # The layer also reduces the number of channels by a factor of 2.
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        """
        x: our feature map
        encFeatures: the list of intermediate outputs from the encoder
        """
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # Upsample the input to our decoder by passing it through our i-th upsampling block
            x = self.upconvs[i](x)

            # Since we have to concatenate (along the channel dimension) the i-th intermediate
            # feature map from the encoder (encFeatures[i]) with our current output x
            # from the upsampling block, we need to ensure that the spatial dimensions of encFeatures[i] and x match.
            # So crop the current features from the encoder blocks.
            encFeat = self.crop(encFeatures[i], x)

            # Concatenate our cropped encoder feature maps (encFeat) with our current upsampled
            # feature map x, along the channel dimension
            x = torch.cat([x, encFeat], dim=1)

            # Pass the concatenated output through our i-th decoder block
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x

    def crop(self, encFeatures, x):
        """
        Take an intermediate feature map from the encoder (encFeatures)
        and a feature map output from the decoder (x), and spatially
        crop the former to the dimension of the latter.
        """
        # Grab the spatial dimensions of x
        (_, _, H, W) = x.shape

        # Crop encFeatures to the spatial dimension [H, W] using CenterCrop
        encFeatures = CenterCrop([H, W])(encFeatures)

        # return the cropped output (features)
        return encFeatures


class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64),
                 decChannels=(64, 32, 16),
                 nbClasses=1, retainDim=True,
                 outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        """
        encChannels: gradual increase in channel dimension as our input passes through the encoder
            We start with 3 channels (RGB) and subsequently [double] the number of channels.

        decChannels: gradual decrease in channel dimension as our input passes through the decoder.
            We reduce the channels by a factor of 2 at every step.

        nbClasses: defines the number of segmentation classes where we have to classify each pixel.
            This usually corresponds to the number of channels in our output segmentation map,
            where we have one channel for each class.
            Since it's binary classification, we keep a single channel and use thresholding for classification

        retainDim: indicates whether we want to retain the original output dimension.

        outSize: spatial dimensions of the output segmentation map.
            We set this to the same dimension as our input image (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
        """
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)

        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])

        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)

        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)

        # return the segmentation map
        return map

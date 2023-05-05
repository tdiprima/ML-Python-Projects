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
        self.enc_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

        # Initialize a MaxPool2d() layer, which reduces the spatial dimension
        # of the feature maps by a factor of 2
        self.pool = MaxPool2d(2)

    def forward(self, x):
        """
        Initialize an empty list to store the intermediate outputs.
        Note that this will enable us to later pass these outputs to our
        decoder, where they can be processed with the decoder feature maps.
        """
        block_outputs = []

        # loop through each block in our encoder
        for block in self.enc_blocks:
            # Process the input feature map through the block
            x = block(x)
            # Add the output of the block to our block_outputs list
            block_outputs.append(x)
            # Apply the max pool operation on our block output
            x = self.pool(x)

        # return the list containing the intermediate outputs
        return block_outputs


class Decoder(Module):
    # The channels gradually decrease by a factor of 2 instead of increasing.

    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels

        # Define a list of upsampling blocks that use the ConvTranspose2d layer
        # to upsample the spatial dimension of the feature maps by a factor of 2.
        # upconvolution 64 to 32, 32 to 16 respectively
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])

        # define 3*3 conv and RELU block
        # The layer also reduces the number of channels by a factor of 2.
        self.dec_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def forward(self, x, enc_features):
        """
        x: our feature map
        enc_features: the list of intermediate outputs from the encoder
        """
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # Upsample the input to our decoder by passing it through our i-th upsampling block
            x = self.upconvs[i](x)

            # Since we have to concatenate (along the channel dimension) the i-th intermediate
            # feature map from the encoder (enc_features[i]) with our current output x
            # from the upsampling block, we need to ensure that the spatial dimensions of enc_features[i] and x match.
            # So crop the current features from the encoder blocks.
            enc_feat = self.crop(enc_features[i], x)  # todo: enc_feat is like "temp"

            # Concatenate our cropped encoder feature maps (enc_feat) with our current upsampled
            # feature map x, along the channel dimension
            x = torch.cat([x, enc_feat], dim=1)

            # Pass the concatenated output through our i-th decoder block
            # 3*3 conv and RELU for each upscending layer.
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x

    def crop(self, enc_features, x):
        """
        Take an intermediate feature map from the encoder (enc_features)
        and a feature map output from the decoder (x), and spatially
        crop the former to the dimension of the latter.
        """
        # Grab the spatial dimensions of x
        (_, _, H, W) = x.shape

        # Crop enc_features to the spatial dimension [H, W] using CenterCrop
        enc_features = CenterCrop([H, W])(enc_features)

        # return the cropped output (features)
        return enc_features


class UNet(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16),
                 nb_classes=1, retain_dim=True,
                 out_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        """
        enc_channels: gradual increase in channel dimension as our input passes through the encoder
            We start with 3 channels (RGB) and subsequently [double] the number of channels.

        dec_channels: gradual decrease in channel dimension as our input passes through the decoder.
            We reduce the channels by a factor of 2 at every step.

        nb_classes: defines the number of segmentation classes where we have to classify each pixel.
            This usually corresponds to the number of channels in our output segmentation map,
            where we have one channel for each class.
            Since it's binary classification, we keep a single channel and use thresholding for classification

        retain_dim: indicates whether we want to retain the original output dimension.

        out_size: spatial dimensions of the output segmentation map.
            We set this to the same dimension as our input image (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
        """
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        # initialize the regression head and store the class variables
        self.head = Conv2d(dec_channels[-1], nb_classes, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        # grab the features from the encoder
        enc_features = self.encoder(x)

        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        dec_features = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])

        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map_ = self.head(dec_features)

        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retain_dim:
            map_ = F.interpolate(map_, self.out_size)

        # return the segmentation map
        return map_

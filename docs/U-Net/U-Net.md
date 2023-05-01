## U-Net 

<span style="color: #0000dd; font-size: larger">U-Net is a convolutional neural network that was developed for biomedical image segmentation<!-- at the Computer Science Department of the University of Freiburg-->.</span>

The architecture diagram of the UNet neural network. It's a series of **convolutional** and **pooling** layers followed by **up-convolutional** layers.

<img src="https://www.mdpi.com/electronics/electronics-11-03755/article_deploy/html/images/electronics-11-03755-g002.png" width="600">

<br>

<img src="https://pub.mdpi-res.com/electronics/electronics-11-03755/article_deploy/html/images/electronics-11-03755-g001.png?1668735123" width="600">

<br>

## U-Net Implementation

See: **unet-torch.py**

A basic U-Net architecture that takes a **3-channel input image** and outputs a **single-channel segmentation mask**.

It consists of a **downsampling** path and an **upsampling** path, with **skip connections** between them.

The **`DoubleConv`** class defines a module that applies two convolutional layers with batch normalization and ReLU activation.

The **`UNet`** class defines the U-Net architecture, with the **`features`** parameter specifying the number of channels for each layer.

The **`downs`** list contains the downsampling blocks.

The **`ups`** list contains the upsampling blocks.

The **`bottleneck`** block is the central block in the network that connects the downsampling and upsampling paths.

The **`forward`** method performs the forward pass of the U-Net network.

The **`skip_connections`** list stores the outputs of the downsampling blocks for later use in the upsampling path.

The **bottleneck block** connects the downsampling and upsampling paths.

The **upsampling blocks** upsample the feature maps and concatenate them with the corresponding skip connection from the downsampling path.

Finally, the **`final_conv`** layer performs the final convolution to produce the segmentation mask.

<br>

# Convolution

In TensorFlow, the equivalent methods for nn.Conv2d and nn.ConvTranspose2d in PyTorch are:

## Conv2D

tf.keras.layers.Conv2D: This layer performs a 2D convolution on the input data.

The parameters of the layer include the number of filters, filter size, stride, padding, and activation function.

This layer is similar to nn.Conv2d in PyTorch.

```py
import torch.nn as nn
self.conv1 = nn.Conv2d(in_channels, hidden_units, kernel_size=3, stride=1, padding=1)

from keras.layers import Conv2D
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

# conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
```

<br>

## Conv2DTranspose

tf.keras.layers.Conv2DTranspose: This layer performs a 2D transposed convolution on the input data.

The parameters of the layer include the number of filters, filter size, stride, padding, and activation function.

This layer is similar to nn.ConvTranspose2d in PyTorch.

```py
self.upconv2 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)

from tensorflow.keras.layers import Conv2DTranspose
# See below.
```

<br>
## Code

Here's an example code snippet that shows how to create a convolutional neural network using tf.keras.layers.Conv2D and tf.keras.layers.Conv2DTranspose:

```python
import tensorflow as tf

# Create a convolutional neural network
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Print the summary of the model
model.summary()
```

<br>
In this example, we create a simple convolutional neural network that consists of a:

1. Conv2D layer, followed by a...
2. MaxPooling2D layer
3. Conv2DTranspose layer
4. UpSampling2D layer
5. Conv2DTranspose layer (yes, another one)

The input shape is `(28, 28, 1)`, which corresponds to a single-channel (grayscale) image with a size of 28x28 pixels. The output shape is also `(28, 28, 1)`...

<br>

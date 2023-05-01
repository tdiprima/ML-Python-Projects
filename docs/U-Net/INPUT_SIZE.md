## Find U-Net Input Size

<mark>**Foolproof method!**</mark>

(But you should cut it off after you get your answer.  Then comment and start again.)

Let's say I have a PyTorch unet model.  When I run the model, is there any way to print out the input size at runtime?

Yes, by accessing the input tensor's **shape** in the **forward method** of your model:

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # define your UNet model here

    def forward(self, x):
        print('Input size:', x.size())
        # rest of the forward method implementation
```

<br>

The **`forward`** method of the `UNet` model includes a print statement that will display the size of the input tensor `x`. The **`size()`** method returns a tuple of integers representing the dimensions of the tensor.

```python
from torchinfo import summary

summary(model,
        input_size=(32, 3, 224, 224),  # make sure this is "input_size" (batch_size, color_channels, height, width), not "input_shape" (int)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
```

## TensorFlow

In a U-Net model implemented in Python, you can find the input size of the model by looking at the **definition of the input layer.** 

The input layer is typically defined using a function. In TensorFlow, the input layer is created using the <mark>**`tf.keras.layers.Input`**</mark> function. 

You can inspect the input layer by looking for the line of code where the input layer is defined. The function call to define the input layer should include the **`input_shape` parameter**, which specifies the shape of the input data. 

For example, the input layer for a U-Net model that takes 256x256 grayscale images as input could be defined like this:

```py
input_layer = tf.keras.layers.Input(shape=(256, 256, 1))
```

In this example, the `shape` parameter of the `Input` function specifies that the input data is 256 pixels wide and 256 pixels tall, and has 1 channel (i.e., it's a grayscale image). 

<br>

## Find U-Net Input Size

To print the input size at runtime, access the input tensor's **shape** in the **forward method** of your model.

(But you should cut it off after you get your answer.  Then comment and start again.)

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

### Model Summary

Now when you go to do your summary, everything is beautiful.

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

### TensorFlow

In a U-Net model implemented in Python, you can find the input size of the model by looking at the **definition of the input layer.** 

The input layer is typically defined using a function. In TensorFlow, the input layer is created using the **`Input`** function.

You can inspect the input layer by looking for the line of code where the input layer is defined. The function call to define the input layer should include the **`input_shape`** parameter, which specifies the shape of the input data. 

For example, the input layer for a U-Net model that takes 256x256 grayscale images as input could be defined like this:

```py
from tensorflow.keras.layers import Input

# shape=(H, W, C)
input_layer = Input(shape=(256, 256, 1))
```

<br>
1 channel (i.e., it's a grayscale image)

<br>

```py
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_shape=1, hidden_units=10, output_shape=10):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_shape)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
```

<br>

In this case, `self.fc1` stands for a fully connected layer that maps the input tensor to a hidden feature space.

A fully connected layer is represented by the `nn.Linear` module.

The `nn.Linear` module takes two arguments as input: 

* number of input features
* number of output features

`self.fc1` is a fully connected layer that maps an **input tensor** with shape `(batch_size, input_shape)` to a **hidden feature space** with `hidden_units` features.

We can think of `self.fc1` as a set of **learnable weights and biases** that transform the input tensor to a different feature space, where the hidden units **capture different aspects** of the input data.

The ReLU activation function is applied after `self.fc1` to **introduce non-linearity** in the model.

After passing through `self.fc1`, the hidden features are then passed through another fully connected layer `self.fc2` with `output_shape` output features, to generate the final output tensor with shape `(batch_size, output_shape)`.

## self.whatever are instance variables

`self.fc1` and `self.fc2` are instance variables of the `MyModel` class.

The `self` keyword refers to the instance of the class.

`fc1` and `fc2` are properties of that instance.

Fully connected layers are **implemented as modules,** which are a type of object that defines a computation graph for a neural network.

The `nn.Module` class provides the functionality to **define and track the state** of these modules, such as the learnable parameters of each layer.

By assigning fully connected layers to `self.fc1` and `self.fc2`, we are defining these modules as attributes of the `MyModel` class instance.

We can then use these modules in the `forward()` method to define the computation graph for our neural network.

## conv2d

`conv2d` is a **2-dimensional convolution** operation, which operates on tensors with **3 or 4** dimensions.

The **expected input format** for `conv2d` is `[batch_size, channels, height, width]` for a batched input, or `[channels, height, width]` for an unbatched input.

Based on the error message, it seems that your model is expecting an unbatched input with shape `[channels, height, width]`, but you are passing in a tensor with shape `[1, 1]`.

One way to fix this error is to reshape your input tensor to have the correct shape before passing it to your model.

This code doesn't shape it; it just makes it the right shape from the get-go:

```py
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_units=10, output_shape=10):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], hidden_units, kernel_size=3)
        self.fc1 = nn.Linear(hidden_units * 26 * 26, output_shape)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# create model
model = MyModel()

# create dummy tensor with shape [1, 1, 28, 28]
input_tensor = torch.randn(1, 1, 28, 28)

# pass through model and check output shape
output_tensor = model(input_tensor)
assert output_tensor.shape == (1, 10)

```

<br>

In this example, we reshape the input tensor to have shape `[1, 1, 28, 28]` to match the expected input shape for our model, which has an input shape of `(1, 28, 28)`.

We then pass the input tensor through the model as before, and check that the output tensor shape 
is `(1, 10)` as expected.

## What the fluff is this?

```py
x.view(x.size(0), -1)
```

A tensor is a multi-dimensional array of numbers that can represent data 
for input or output of a model.

The `view()` method is used to change the shape of a tensor without changing its underlying data.

The `size() `method returns the size of a tensor along a particular dimension.

In the line `x.view(x.size(0), -1)`, `x` is a tensor that represents the output of a convolutional layer in the model.

The `size(0)` method returns the size of the tensor along the first dimension, 
which corresponds to the batch size.

The `-1` in the second argument to `view()` means that PyTorch will automatically compute 
the size of the second dimension, to be whatever is needed, to preserve the total number of elements in the tensor.

By calling `view()` with the first argument set to `x.size(0)` and the second argument set to `-1`, 
we are reshaping the tensor so that it has two dimensions:

* the batch dimension (which has size `x.size(0)`) 
* and a second dimension that is automatically computed by PyTorch to preserve the total number of elements in the tensor

This reshaped tensor, is then passed to the fully connected layer in the model.

In summary, the line `x.view(x.size(0), -1)` is used to **reshape the output of a convolutional layer** 
so that it can be passed as input to a fully connected layer.

It ensures that the <mark>**batch dimension is preserved** while flattening the remaining dimensions of the tensor.</mark>

## x.size(dim)

`x.size(0)` is the size of the tensor `x` along its first dimension.

In other words, it gives you the **number of elements** in the batch dimension of the tensor.

You could say "the size of `x` along the first dimension",
or "the number of elements in the batch dimension of `x`".

<br>

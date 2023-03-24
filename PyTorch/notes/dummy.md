# Test PyTorch Model 🏵️

If I've created a PyTorch class that subclasses `nn.Module`, where the `input_shape=1`, the `hidden_units=10`, and the
`output_shape=10`, how can I create a dummy (random) tensor, and test to see if the model works?

You can create a dummy tensor using the PyTorch `torch.randn()` function 
and test the model by passing this tensor through the model 
and checking if it produces the expected output shape.

Here is an example code snippet:

```ruby
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

# create model
model = MyModel()

# create dummy tensor
input_tensor = torch.randn(1, 1)

# pass through model and check output shape
output_tensor = model(input_tensor)
assert output_tensor.shape == (1, 10)
```

In this example, we create a `MyModel` class that subclasses `nn.Module`, 
with an input shape of 1, hidden units of 10, and an output shape of 10.

We then create a model instance, and a dummy input tensor using `torch.randn()`.

Finally, we pass the input tensor through the model's `forward()` method to get the output tensor,
and check that its shape matches the expected output shape of `(1, 10)` using an `assertion` statement.

### If it doesn't print any output, that means the assertion worked.

Yes, that is correct. If the assertion statement passes without raising any errors,
then it means that the output tensor shape is as expected, and 
your PyTorch model is working as expected for this particular input.

If the assertion statement fails, it means that the output tensor shape is not as expected,
and you may need to debug your model implementation to see what went wrong.

# self.fc1

In the example code, `self.fc1` stands for a fully connected layer that maps 
the input tensor to a hidden feature space.

In PyTorch, a **fully connected layer** is represented by the `nn.Linear` module.

The `nn.Linear` module takes two arguments as input: 
the number of input features, and the number of output features.

In this case, `self.fc1` is a fully connected layer that maps an input tensor 
with shape `(batch_size, input_shape)` to a hidden feature space with `hidden_units` features.

We can think of `self.fc1` as a set of learnable weights and biases that transform
the input tensor to a different feature space, where the hidden units capture
different aspects of the input data.

<mark>The **ReLU** activation function is applied after `self.fc1`
to **introduce non-linearity** in the model.</mark>

After passing through `self.fc1`, the hidden features are then passed through 
another fully connected layer `self.fc2` with `output_shape` output features,
to generate the final output tensor with shape `(batch_size, output_shape)`.

### self.fcn are instance variables of this class

In Python, `self.fc1` and `self.fc2` are **instance variables** of the MyModel class.

When we create an instance of the MyModel class, these variables are initialized 
as fully connected layers with the specified input and output dimensions.

The `self` keyword refers to the **instance of the class**.

`fc1` and `fc2` are **properties** of that instance.

In PyTorch, fully connected layers are implemented as **modules**, which are 
a type of object that defines a computation graph for a neural network.

The `nn.Module` class provides the functionality to define and track 
the state of these modules, such as the learnable parameters of each layer.

By assigning fully connected layers to `self.fc1` and `self.fc2`, we are defining 
these **modules as attributes** of the MyModel class instance.

We can then use these modules in the `forward()` method to define 
the computation graph for our neural network.

# Fix runtime error

(You guessed it, a shape error.)

> "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [1, 1]"?

This error occurs because your model is expecting an input tensor with either 3 or 4 dimensions,
but you are passing in a tensor with 2 dimensions.

The error message suggests that the problem is related to a `conv2d` operation in your model.

`conv2d` is a 2-dimensional convolution operation, which operates on tensors with 3 or 4 dimensions.

The expected input format for `conv2d` is `[batch_size, channels, height, width]` 
for a batched input, or `[channels, height, width]` for an unbatched input.

Based on the error message, it seems that your model is expecting an unbatched input 
with shape `[channels, height, width]`, but you are passing in a tensor with shape `[1, 1]`.

One way to fix this error is to **reshape** your input tensor to have the correct shape 
before passing it to your model.

Here's an example code snippet that demonstrates how to reshape a tensor to have the correct shape:

```ruby
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

In this example, we reshape the input tensor to have shape `[1, 1, 28, 28]` to match the expected input shape for our model, 
which has an input shape of `(1, 28, 28)`.

We then pass the input tensor through the model as before, and check that the output tensor shape 
is `(1, 10)` as expected.

# x.view(x.size(0), -1)

In PyTorch, a tensor is a multi-dimensional array of numbers that can represent data 
for input or output of a model.

The `view()` method is used to change the shape of a tensor without changing its underlying data.

The `size() `method returns the size of a tensor along a particular dimension.

In the line `x.view(x.size(0), -1)`, `x` is a tensor that represents the output of a convolutional layer in the model.

The `size(0)` method returns the size of the tensor along the first dimension, 
which corresponds to the batch size.

The `-1` in the second argument to `view()` means that PyTorch will **automatically compute** 
the size of the second dimension, to be whatever is needed, to preserve the total number of elements in the tensor.

By calling `view()` with the first argument set to `x.size(0)` and the second argument set to `-1`, 
we are **reshaping the tensor** so that it has two dimensions:

* the batch dimension (which has size `x.size(0)`) 
* and a second dimension that is automatically computed by PyTorch to **preserve the total number of elements** in the tensor

This reshaped tensor, is then passed to the fully connected layer in the model.

**In summary,** the line `x.view(x.size(0), -1)` is used to reshape the output of a convolutional layer 
so that it can be passed as input to a fully connected layer.

It ensures that the <mark>**batch dimension is preserved while flattening the remaining dimensions of the tensor.**</mark>

### How would I say "x.size(0)" in English?

I think you would just say, "x size zero", or "x size of 0", as in, the zeroth dimension.

`x.size(0)` is the size of the tensor `x` along its first dimension.

In other words, it gives you the number of elements in the batch dimension of the tensor.

You could say "the size of `x` along the first dimension",
or "the number of elements in the batch dimension of `x`".

<br>

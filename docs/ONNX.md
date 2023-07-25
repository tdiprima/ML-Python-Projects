# ONNX

Open Neural Network Exchange.

So, you know how you can play a game on your PlayStation, but you can't play that same game on your Xbox or Nintendo without buying a new version? That's because they're all different systems and they can't understand each other's language. 

**ONNX** is like a **universal translator** for machine learning models</mark>, which are like the "games" in this analogy.

It lets you train a model (or create a game) in one system, like PyTorch (PlayStation), and then use it in another system, like TensorFlow (Xbox), without having to start over from scratch. This makes it a lot easier for scientists and developers to use and share their work. 

Just as a game can be converted from PlayStation to Xbox, ONNX helps in **converting machine learning models** from one framework to another. And that's really cool!

## Example

Let's take an example using PyTorch and ONNX. Imagine you have a simple PyTorch model and you want to convert it to ONNX format. Here is how you could do that:

First, we'll define a simple model in PyTorch:

```python
import torch
import torch.nn as nn

# Here's a simple model with one linear layer
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# Create the model with input and output dimensions
model = SimpleModel(3, 2)
```

<br>
Now, let's create a dummy input to run the model and get the output:

```python
# Creating a dummy input
dummy_input = torch.randn(10, 3)
```

<br>

To convert the PyTorch model to ONNX, we can use the `torch.onnx.export` function:

```python
# Export the model
torch.onnx.export(model,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  "simple_model.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```

<br>

The `simple_model.onnx` file is a binary protobuf file which contains both the network structure and parameters of the model we exported (in this case, it is `model`).

**That's it!** You now have an ONNX model (`simple_model.onnx`) which can be loaded in another deep learning framework that supports ONNX for inference.

Please note that you need to have the 'onnx', 'torch', and 'torchvision' libraries installed for the code to run. You can install them using pip:

```sh
pip install onnx torch torchvision
```

You should adjust these commands according to your environment and needs, and it's generally recommended to do such installations in a virtual environment to avoid conflicts with other packages.

<br>

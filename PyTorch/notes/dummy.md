## Test PyTorch Model 🏵️

How can I create a dummy (random) tensor, and test to see if the model works?

Create a dummy tensor using the PyTorch `torch.randn()` function.
 
Test the model by passing this tensor through the model.
 
Check if it produces the expected output shape.

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


# create model
model = MyModel()

# create dummy tensor
input_tensor = torch.randn(1, 1)  # [[0.6867]], size [1, 1]

# pass through model and check output shape
output_tensor = model(input_tensor)

# output tensor([[-0.2182, -0.2669, -0.0844,  0.0432, -0.2706, -0.1959, -0.1091,  0.6967, -0.3101, -0.3597]], grad_fn=<AddmmBackward0>), size [1, 10]

assert output_tensor.shape == (1, 10)

print("\nIf there are no errors, it succeeded.")

```

<br>

We pass the input tensor through the model's `forward()` method to get the output tensor, and check that its shape matches the expected output shape of `(1, 10)` using an `assertion` statement.

**No news is good news.**

However, if the assertion statement fails, it means that the output tensor shape is not as expected,
and you may need to debug your model implementation to see what went wrong.

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
print("\noutput_tensor shape:", output_tensor.shape)

# ===============================

import torch
import torch.nn as nn


class MyModel1(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_units=10, output_shape=10):
        super(MyModel1, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], hidden_units, kernel_size=3)
        self.fc1 = nn.Linear(hidden_units * 26 * 26, output_shape)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# create model
model = MyModel1()

# create dummy tensor with shape [1, 1, 28, 28]
input_tensor = torch.randn(1, 1, 28, 28)

# pass through model and check output shape
output_tensor = model(input_tensor)
assert output_tensor.shape == (1, 10)
print("\noutput_tensor shape:", output_tensor.shape)

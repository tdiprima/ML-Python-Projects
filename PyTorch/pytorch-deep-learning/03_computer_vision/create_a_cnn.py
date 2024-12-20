"""
Creates a dummy tensor to test the output shape of a FashionMNISTModel.
"""
import sys

import torch
from torchinfo import summary

sys.path.append('../toolbox')
from my_models import FashionMNISTModelV2 as FashionMNISTModel

model = FashionMNISTModel(input_shape=1, hidden_units=10, output_shape=10)

# create dummy tensor
input_tensor = torch.randn(1, 1, 28, 28)

# pass through model and check output shape
output_tensor = model(input_tensor)
assert output_tensor.shape == (1, 10)
print(f"\noutput_tensor shape: {output_tensor.shape}")

summary(model, input_size=[1, 1, 28, 28])

"""
1 x.shape: torch.Size([1, 10, 14, 14])
2 x.shape: torch.Size([1, 10, 7, 7])
3 x.shape: torch.Size([1, 10])
torch.Size([1, 10])
"""

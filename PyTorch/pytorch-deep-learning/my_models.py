import torch
from torch import nn


# Define your model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        # super().__init__()

        super(LinearRegressionModel, self).__init__()
        # define your model layers here
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    # Forward = computation
    # "x" is the input data (e.g. training/testing features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linear regression formula (y = m*x + b)
        return self.weights * x + self.bias

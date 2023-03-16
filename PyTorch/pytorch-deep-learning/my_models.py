import torch
from torch import nn  # nn contains all of PyTorch's building blocks for neural networks


class LinearRegressionModel(nn.Module):
    """
    Create a Linear Regression model class
    Almost everything in PyTorch is a nn.Module (like neural network lego blocks)
    Start with random weights and bias (this will get adjusted as the model learns)
    requires_grad=True: Can we update this value with gradient descent?
    PyTorch loves float32.
    """

    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True
        )

        self.bias = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True
        )

    """
    Forward defines the computation in the model
    "x" is the input data (e.g. training/testing features)
    This is the linear regression formula (y = m*x + b)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# SUBCLASS nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # CREATE LAYERS
        self.layer_1 = nn.Linear(in_features=2, out_features=5)  # takes in 2 features and up-scales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
        # out_features=1 takes in 5 features from previous layer and outputs a single feature (same shape as y)

    # CREATE forward() METHOD
    def forward(self, x):
        return self.layer_2(self.layer_1(x))  # x -> layer_1 ->  layer_2 -> output

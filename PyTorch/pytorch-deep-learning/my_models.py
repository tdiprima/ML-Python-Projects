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

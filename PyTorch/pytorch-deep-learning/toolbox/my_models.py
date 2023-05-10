import torch
from torch import nn  # nn contains all of PyTorch's building blocks for neural networks


class LinearRegressionModel(nn.Module):
    """
    A "hello world" Linear Regression model class
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
        # print("\nInput size:", x.size())
        return self.weights * x + self.bias


class CircleModelV0(nn.Module):
    """
    Model for the "make_circles" dataset from sklearn.datasets
    """

    def __init__(self):
        super().__init__()
        # We said our data was "non-linear", but we're creating linear layers.
        self.layer_1 = nn.Linear(in_features=2, out_features=5)  # takes in 2 features and up-scales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
        # out_features=1 takes in 5 features from previous layer and outputs a single feature (same shape as y)

    # Define a forward method containing the forward pass computation
    def forward(self, x):
        # print("\nInput size:", x.size())
        # x = self.layer_1(x)
        # print("layer_1", x.size())
        # x = self.layer_2(x)
        # print("layer_2", x.size())
        # return x
        return self.layer_2(self.layer_1(x))  # x -> layer_1 ->  layer_2 -> output

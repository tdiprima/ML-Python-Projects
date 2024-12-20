"""
Defines several classes implementing various machine learning models, including linear regression, multi-layer perceptron
for circle dataset, and convolutional neural network for FashionMNIST dataset.
Each class features an initializer for setting up the layers and a forward method for performing the forward pass calculation.
"""
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

        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    """
    Forward defines the computation in the model
    "x" is the input data (e.g. training/testing features)
    This is the linear regression formula (y = m*x + b)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("\nInput size:", x.size())
        return self.weights * x + self.bias


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("\nInput size:", x.size())
        return self.linear_layer(x)


class CircleModelV0(nn.Module):
    """
    Model for the "make_circles" dataset from sklearn.datasets
    Two layers
    """

    def __init__(self):
        super().__init__()
        # We said our data was "non-linear", but we're creating linear layers.
        self.layer_1 = nn.Linear(in_features=2, out_features=5)  # takes in 2 features and up-scales to 5 features
        self.layer_2 = nn.Linear(in_features=5,
                                 out_features=1)  # out_features=1 takes in 5 features from previous layer and outputs a single feature (same shape as y)

    # Define a forward method containing the forward pass computation
    def forward(self, x):
        # print("\nInput size:", x.size())
        # x = self.layer_1(x)
        # print("layer_1", x.size())
        # x = self.layer_2(x)
        # print("layer_2", x.size())
        # return x
        return self.layer_2(self.layer_1(x))  # x -> layer_1 ->  layer_2 -> output


# Create a model
class CircleModelV1(nn.Module):
    """
    Three layers
    """

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        # print("\nInput size:", x.size())
        return self.layer_3(self.layer_2(self.layer_1(x)))


class CircleModelV2(nn.Module):
    """
    Build a model with non-linear activation functions
    """

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # relu is a non-linear activation function

    def forward(self, x):
        # print("\nInput size:", x.size())
        # Put non-linear activation function IN-BETWEEN our layers.
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


class FashionMNISTModelV0(nn.Module):
    """
    This model sux
    """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        # print("\nInput size:", x.size())
        return self.layer_stack(x)


class FashionMNISTModelV1(nn.Module):
    """
    Building a better model with non-linearity
    This model still sux
    Create a model with non-linear and linear layers
    """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # flatten inputs into a single vector
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        # print("\nInput size:", x.size())
        return self.layer_stack(x)


class FashionMNISTModelV2(nn.Module):
    """
    Create a convolutional neural network
    Model architecture that replicates the TinyVGG
    model from CNN explainer website.
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        """
        Create NN in a couple of blocks
        Create Layers inside nn.Sequential
        First 2 layers are feature extractors (learning patterns that best represent our data).
        Last layer (output layer) is classifier layer (classify features into our target classes).
        """

        self.conv_block_1 = nn.Sequential(
            # Create a conv layer - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # values we can set ourselves in our NN's are called hyperparameters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten to single feature vector
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape
                      )  # out: The length of how many classes we have. One value for each class.
        )

    def forward(self, x):
        # print("\nInput size:", x.size())
        x = self.conv_block_1(x)
        # print(f"\nOutput shape of conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape of conv_block_2: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape of classifier: {x.shape}")
        return x

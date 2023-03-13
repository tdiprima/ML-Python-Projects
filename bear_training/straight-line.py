"""
How about we start with a straight line?
And we see if we can build a PyTorch model that learns the pattern of the straight line and matches it.
https://github.com/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb
"""
import torch
from torch import nn
import matplotlib.pyplot as plt

# Check PyTorch version
print("Torch version:", torch.__version__)

# Linear regression formula
# y = a + bX
weight = 0.7  # b (slope)
bias = 0.3  # a (y intercept)

# Create data
start = 0
end = 1
step = 0.02
# Capital = matrix; lowercase = vector
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create train/test split
train_split = int(0.8 * len(X))  # 80% of data used for training set, 20% for testing

# X[:train_split] means: Get all of the examples up until the train split.
X_train, y_train = X[:train_split], y[:train_split]

# X[train_split:] means: Get everything from the train split, onwards.
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    # plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(train_data, train_labels, c="b", s=16, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=16, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=16, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

    plt.show()


# Create a Linear Regression model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    # Forward defines the computation in the model
    # "x" is the input data (e.g. training/testing features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linear regression formula (y = m*x + b)
        return self.weights * x + self.bias


# Set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
list(model_0.parameters())

# List named parameters
model_0.state_dict()

# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

# Note: in older PyTorch code you might also see torch.no_grad()
# with torch.no_grad():
#     y_preds = model_0(X_test)

# Check the predictions
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

plot_predictions(predictions=y_preds)

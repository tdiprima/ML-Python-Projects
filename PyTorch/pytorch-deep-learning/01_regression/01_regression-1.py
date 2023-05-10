"""
Build a PyTorch model that learns the pattern of a straight line and matches it.
"""
import sys

import torch
from torchinfo import summary

sys.path.append('../toolbox')
from my_models import LinearRegressionModel
from plotting import plot_predictions

# Check PyTorch version
print("Torch version:", torch.__version__)

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
# print(X[:10], y[:10])

# Create train/test split
pos = int(0.8 * len(X))  # 80% of data used for training set, 20% for testing
X_train, y_train = X[:pos], y[:pos]
X_test, y_test = X[pos:], y[pos:]
# print("\nlengths:", len(X_train), len(y_train), len(X_test), len(y_test))

# Set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model = LinearRegressionModel()

# SAVE
# torch.save(model.state_dict(), "models/my_model.pth")

# Check the nn.Parameter(s) within the nn.Module subclass we created
print("\nmodel parameters:", list(model.parameters()))

# List named parameters
print("\nmodel dict:", model.state_dict())

# Make predictions with model
with torch.inference_mode():
    y_preds = model(X_test)

# Note: "inference_mode" is "torch.no_grad++"
# with torch.no_grad():
#   y_preds = model(X_test)

# Check the predictions
# print(f"\nNumber of testing samples: {len(X_test)}")
# print(f"Number of predictions made: {len(y_preds)}")
# print(f"\nPredicted values:\n{y_preds}")
# print("\nHow close were we?", y_test - y_preds)

plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)

summary(model, input_size=[10, 1])

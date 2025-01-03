"""
Creates, trains, and tests a simple neural network to fit a straight line to noisy circular data,
using pytorch and sklearn, and plots the results.

SEQUENTIAL
model = nn.Sequential
Basically, we didn't need to do all that (with the class); all we had to do is this.
"""
import sys

import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

sys.path.append('../toolbox')
from plotting import plot_predictions

epochs = 1000

# Create circles
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(len(X_train), len(X_test), len(y_train), len(y_test))  # 800 200 800 200

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# 5.2 Adjusting model to fit a straight line
model = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

# print("\nState dict:", model.state_dict())  # weights and biases

# Loss and optimizer
loss_fn = nn.L1Loss()  # MAE loss with regression data
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Train the model
torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# TODO: TROUBLESHOOTING
# 5.1 Preparing data to see if our model can fit a straight line
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias  # Linear regression formula (without epsilon)

# Check the data
print("\nlen X_regression:", len(X_regression))
# print("\nX and y:", X_regression[:5], y_regression[:5])

# Create train and test splits
train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Check the lengths of each
len(X_train_regression), len(X_test_regression), len(y_train_regression), len(y_test_regression)

plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression)
# END "TROUBLESHOOTING"

# Put the data on the target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

# Training
for epoch in range(epochs):
    y_pred = model(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# Turn on evaluation mode
model.eval()

# Make predictions (inference)
with torch.inference_mode():
    y_preds = model(X_test_regression)

# Plot data and predictions
plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu())

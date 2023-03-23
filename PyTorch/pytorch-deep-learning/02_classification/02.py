"""
CircleModelV0 model_0
2.1 Setup loss function and optimizer
"""
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

# TODO: Num samples
# n_samples = 256
n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# Create a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # https://playground.tensorflow.org/
        # Because we said our data isn't linear, should use nn.ReLU or nn.Tanh.
        # But the parameters aren't the same and IDK how to use it yet, so.
        self.layer_1 = nn.Linear(in_features=2, out_features=8)  # takes in 2 features (X), produces 8 features
        self.layer_2 = nn.Linear(in_features=8, out_features=1)  # takes in 8 features, produces 1 feature (y)

    # Define a forward method containing the forward pass computation
    def forward(self, x):
        return self.layer_2(self.layer_1(x))


# Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)

# Create a loss function
loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


def accuracy_fn(y_true, y_pred):
    """
    Out of 100 examples, what percentage does our model get right?
    Accuracy = True Positive / (True Positive + True Negative) * 100
    sklearn.metrics.accuracy_score(y_true, y_pred)
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# 3.2 Building a training and testing loop
torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# TODO: Num epochs
# epochs = 100
epochs = 1000

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    # Training
    model_0.train()

    # Forward pass (model outputs raw logits)
    # Squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # turn logits -> pred probs -> pred labls

    # Calculate loss/accuracy
    # You need torch.sigmoid() when using nn.BCELoss
    # loss = loss_fn(torch.sigmoid(y_logits), y_train)

    # Using nn.BCEWithLogitsLoss works with raw logits
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()
    # print("HERE")

    # Loss backwards
    loss.backward()
    # print("THERE")

    # Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        # Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # Calculate loss/accuracy
        # PyTorch loss_fn(y_pred, y)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print out what's happening every 10 epochs
    # if epoch % 10 == 0:
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Train Loss: {loss:.5f}, Train Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

import matplotlib.pyplot as plt
import sys

sys.path.append('../toolbox')
from helper_functions import plot_decision_boundary

# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

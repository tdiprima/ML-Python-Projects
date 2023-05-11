"""
CircleModelV0 model
Loss function, optimizer, training
"""
import sys

import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn
from torchinfo import summary

sys.path.append('../toolbox')
from my_models import CircleModelV0
from helper_functions import accuracy_fn

n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(len(X_train), len(X_test), len(y_train), len(y_test))  # 800 200 800 200

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create an instance of the model and send it to target device
model = CircleModelV0().to(device)

# Create a loss function
loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)


# 3.2 Building a training and testing loop
torch.manual_seed(42)
# torch.cuda.manual_seed(42)

epochs = 1000

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    # Training
    model.train()

    # Forward pass (model outputs raw logits)
    # Squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # turn logits -> pred probs -> pred labls

    # Calculate loss/accuracy
    # You need torch.sigmoid() when using nn.BCELoss
    # loss = loss_fn(torch.sigmoid(y_logits), y_train)

    # Using nn.BCEWithLogitsLoss works with raw logits
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backwards
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model.eval()
    with torch.inference_mode():
        # Forward pass
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # Calculate loss/accuracy
        # PyTorch loss_fn(y_pred, y)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print out what's happening every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train Loss: {loss:.5f}, Train Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

import matplotlib.pyplot as plt

sys.path.append('../toolbox')
from helper_functions import plot_decision_boundary

# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")

plot_decision_boundary(model, X_test, y_test)

print()
summary(model, input_size=[800, 2])

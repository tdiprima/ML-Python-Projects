# CircleModelV2 model_3
# 6. The missing piece: non-linearity
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

# TODO: Num samples
# n_samples = 256
n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Convert data to tensors and then to train and test splits

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))
# X_train[:5], y_train[:5]

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# Build a model with non-linear activation functions
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # relu is a non-linear activation function

    def forward(self, x):
        # Put non-linear activation function IN-BETWEEN our layers.
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model_3 = CircleModelV2().to(device)
# model_3

# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.1)

print(len(X_test), len(y_test))


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


# Random seeds
torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# TODO: num epochs
# epochs = 100
epochs = 1000

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Loop through data
for epoch in range(epochs):
    # Training
    model_3.train()

    # Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> prediction probabilities -> prediction labels

    # Calculate the loss
    loss = loss_fn(y_logits, y_train)  # BCEWithLogitsLoss (takes in logits as first input)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()
    # print("HERE")

    # Loss backward
    loss.backward()
    # print("THERE")

    # Step the optimizer
    optimizer.step()

    # Testing
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # TODO: modulus
    if epoch % 100 == 0:
        # if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# Make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

print("\npredictions:", y_preds[:10])
print("\ny_test:", y_test[:10])

import matplotlib.pyplot as plt
import sys
sys.path.append('../toolbox')
from helper_functions import plot_decision_boundary

# Plot decision boundaries
plt.figure(figsize=(6, 6))
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)  # model_3 = has non-linearity

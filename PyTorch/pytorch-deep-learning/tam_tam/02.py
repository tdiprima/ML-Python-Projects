"""
Reduce sample size, it works.
Results: Accuracy doesn't go above 50%
2.1 Setup loss function and optimizer
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
"""

import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

# TODO: Make 1000 samples
# n_samples = 256
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,  # a little bit of noise to the dots
                    random_state=42)  # keep random state so we get the same values

# Turn data into tensors
# Otherwise this causes issues with computations later on
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,  # 20% test, 80% train
                                                    random_state=42)  # make the random split reproducible

print(len(X_train), len(X_test), len(y_train), len(y_test))

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5)  # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1)  # takes in 5 features, produces 1 feature (y)

    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(
            self.layer_1(x))  # computation goes through layer_1 first then the output of layer_1 goes through layer_2


# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)

# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


# Calculate accuracy (a classification metric)
# Out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


# 3.1 Going from raw logits -> prediction probabilities -> prediction labels
# View the first 5 outputs of the forward pass on the test data
# TODO: WAIT, WHAT?
# y_logits = model_0(X_test.to(device))[:5]
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]

# Use sigmoid on model logits => prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)

# Find the predicted labels (and round the prediction probabilities)
y_preds = torch.round(y_pred_probs)

# In full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
y_preds.squeeze()
# EVERYTHING GOOD UP TO HERE

# 3.2 Building a training and testing loop
torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# Set the number of epochs
# epochs = 100
epochs = 1000

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    # Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # turn logits -> pred probs -> pred labls

    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train)
    loss = loss_fn(y_logits,  # Using nn.BCEWithLogitsLoss works with raw logits
                   y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()
    # print("HERE")

    # 4. Loss backwards
    loss.backward()
    # print("THERE")

    # 5. Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Print out what's happening every 10 epochs
    # if epoch % 10 == 0:
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        # THIS SHOULD BE FINE.  IT'S NOT (with 1000 samples).

import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

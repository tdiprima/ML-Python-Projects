"""
CircleModelV0 model (2 layers)
Untrained, predictions are negative numbers.
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torchinfo import summary

sys.path.append('../toolbox')
from my_models import CircleModelV0

# Create circles
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)
# X.shape (1000, 2); y.shape (1000,)

# Make DataFrame of circle data
# This is just a dictionary
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
# print("DataFrame:", circles.head(10))

# What's it look like?
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

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

# Make predictions with the UNTRAINED model
with torch.inference_mode():
    untrained_preds = model(X_test.to(device))

print(f"\nLength of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
# print(f"\nFirst 10 predictions:\n{torch.round(untrained_preds[:10])}")  # ROUNDED
# print(f"\nFirst 10 test labels:\n{y_test[:10]}")

print()
summary(model, input_size=[200, 2])

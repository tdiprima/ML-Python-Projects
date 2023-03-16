# import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
# from torch import nn

from my_models import CircleModelV0

# Make 1000 samples
n_samples = 1000

"""
Make-circles dataset
Make a large circle containing a smaller circle in 2d.
A simple toy dataset to visualize clustering and classification algorithms.
"""
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

"""
Insert a dictionary.
We'll call the features in the 1st column, X1.
2nd column, X2.
"""
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

# Visualize, visualize, visualize
# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
#
# plt.show()

# 1. GET DATA READY. TURN INTO TENSORS.

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]

print(f"\nValues for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

print("\nBefore:", type(X), X.dtype, y.dtype)

"""
Turn numpy data into tensors
numpy = float64
tensor = float32
"""
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print("After:", type(X), X.dtype, y.dtype)

"""
SPLIT!
Training & testing FEATURES; training & testing LABELS.
"""
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,  # 0.2 = 20% test; 80% train.
                                                    random_state=42)  # Like seed.

# 2. BUILD A MODEL TO CLASSIFY BLUE / RED.

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\ndevice:", device)

# INSTANTIATE
model_0 = CircleModelV0().to(device)

print("\nmodel_0", model_0)

# Let's replicate the model above using nn.Sequential()
# model_0 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=5, out_features=1)
# ).to(device)

# MAKE PREDICTIONS
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

print(f"\nLength of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{torch.round(untrained_preds[:10])}")
print(f"\nFirst 10 labels:\n{y_test[:10]}")

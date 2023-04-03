import sys

from torch import nn

# Creating a toy multi-class dataset
import torch
import matplotlib.pyplot as plt
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,  # give the clusters a little shake up
                            random_state=RANDOM_SEED)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# 3. Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# 4. Plot data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

# BUILDING A MULTI-CLASS CLASSIFICATION MODEL
# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# Build a multi-class classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """
        Initializes multi-class classification model.

        Args:
          input_features (int): Number of input features to the model
          output_features (int): Number of outputs features (number of output classes)
          hidden_units (int): Number of hidden units between layers, default 8
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


# Create an instance of BlobModel and send it to the target device
model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)

# X_blob_train.shape, y_blob_train[:5]

# print("\nunique y_blob train:\n", torch.unique(y_blob_train))

# Create a loss function for multi-class classification
# Loss function measures how wrong our model's predictions are
from torch import nn

loss_fn = nn.CrossEntropyLoss()

# Create an optimizer for multi-class classification
# Optimizer updates our model parameters to try and reduce the loss
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)  # learning rate is a hyperparameter you can change

"""
GETTING PREDICTION PROBABILITIES
"""

# LOGITS: Let's get some raw outputs of our model
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device))

# y_logits[:10]

# y_blob_test[:10]

# SOFTMAX: Convert our model's logit outputs to prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(f"\ny_logits[:5]\n {y_logits[:5]}")
print(f"\ny_pred_probs[:5]\n{y_pred_probs[:5]}\n")

# Convert our model's prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)
print("\ny_preds:\n", y_preds)
# y_blob_test
# y_blob_train.dtype

# Creating a training loop and testing loop
sys.path.append('../toolbox')
from helper_functions import accuracy_fn, plot_decision_boundary

# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
epochs = 100

# Put data to the target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

# Loop through data
for epoch in range(epochs):
    # TRAINING
    model_4.train()

    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TESTING
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_preds)

    # PRINT
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%")

# MAKING AND EVALUATING PREDICTIONS

# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# View the first 10 predictions
# y_logits[:10]

# Go from logits -> Prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
# print("\nPred probs:\n", y_pred_probs[:10])

# Go from pred probs to pred labels
y_preds = torch.argmax(y_pred_probs, dim=1)
# print("\nPred labels:\n", y_preds[:10])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)

from torchmetrics import Accuracy

try:
    # Setup metric
    torchmetric_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

    # Calculate accuracy
    metric = torchmetric_accuracy(y_preds, y_blob_test)

    print("\nmetric:", metric)
    print("\ntorchmetric_accuracy:", torchmetric_accuracy)
    print("\ndevice:", torchmetric_accuracy.device)
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("HERE.", exc_type, exc_obj, exc_tb.tb_lineno)
    sys.exit(1)

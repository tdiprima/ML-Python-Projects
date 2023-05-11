"""
Model/tensors to(device)
"""
import sys

import torch
from torch import nn
from torchinfo import summary

sys.path.append('../toolbox')
from my_models import LinearRegressionModelV2
from plotting import plot_predictions

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 1000

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create weight and bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
# print(X[:10], y[:10])

# Split data
split_pos = int(0.8 * len(X))
X_train, y_train = X[:split_pos], y[:split_pos]
X_test, y_test = X[split_pos:], y[split_pos:]
# print(len(X_train), len(y_train), len(X_test), len(y_test))

plot_predictions(X_train, y_train, X_test, y_test)

"""
Set the manual seed when creating the model (this isn't always needed,
but try commenting it out and seeing what happens)
"""
torch.manual_seed(42)
model = LinearRegressionModelV2()
# model, model.state_dict()

# Set model to GPU if it's available, otherwise it'll default to CPU
model.to(device)  # the device variable was set above to be "cuda" if available or "cpu" if not

# Create loss function
loss_fn = nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

torch.manual_seed(42)

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    # Training
    model.train()  # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    # Testing
    model.eval()  # put the model in evaluation mode for testing (inference)

    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

# Find our model's learned parameters
from pprint import pprint  # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html

print("\nThe model learned the following values for weights and bias:")
pprint(model.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# Making predictions
# Turn model into evaluation mode
model.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model(X_test)
print("\ny_preds", y_preds)

# Put data on the CPU and plot it
# plot_predictions(predictions=y_preds.cpu())
plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds.cpu())

from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("../models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "pytorch_workflow_model1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"\nSaving model to: {MODEL_SAVE_PATH}")
# Only saving the state_dict() only saves the model's learned parameters
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

# Instantiate a fresh instance of LinearRegressionModelV2
loaded_model_1 = LinearRegressionModelV2()

# Load model state dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
loaded_model_1.to(device)

print(f"\nLoaded model:\n{loaded_model_1}")
print(f"Model on device: {next(loaded_model_1.parameters()).device}")

# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

print("\nEqual?", y_preds == loaded_model_1_preds)

summary(model, input_size=[40, 1])

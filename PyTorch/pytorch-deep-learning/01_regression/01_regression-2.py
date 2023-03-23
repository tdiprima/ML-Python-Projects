import sys

import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.append('../toolbox')

from my_models import LinearRegressionModel

from plotting import plot_predictions

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

split_position = int(0.8 * len(X))
X_train, y_train = X[:split_position], y[:split_position]
X_test, y_test = X[split_position:], y[split_position:]

torch.manual_seed(42)

# LOAD THE PRE-TRAINED MODEL
PATH = "../models/my_model.pth"
model_0 = LinearRegressionModel()
model_0.load_state_dict(torch.load(PATH))

# CONTINUE FROM STEP 1

# Create the loss function
loss_fn = nn.L1Loss()  # Mean absolute error is same as L1Loss

# Create the optimizer
# parameters of target model to optimize
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

"""
The testing loop involves going through the testing data and evaluating how good the patterns are that the model learned on the training data (the model never see's the testing data during training).
Each of these is called a "loop" because we want our model to look (loop through) at each sample in each dataset.
"""

# Training loop
# Train our model for N epochs (forward passes through the data) and we'll evaluate it every 10 epochs.

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
# epochs = 100
epochs = 200  # TODO: See? This is how we get our stuff to converge even more.

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # TRAINING

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    # TESTING

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
        # 1. Forward pass on test data
        test_pred = model_0(X_test)

        # 2. Calculate loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float))
        # predictions come in torch.float, so comparisons need to be done with tensors in torch.float

        # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# Plot the loss curves
# todo: idk why this is all fluffed up now
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

# Find our model's learned parameters
print("\nThe model learned the following values for weights and bias:")
print(model_0.state_dict())

print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# NOW, MAKE PREDICTIONS WITH IT

# 1. Set the model in evaluation mode
model_0.eval()

# 2. Set up the inference mode context manager
with torch.inference_mode():
    # 3. Make sure the calculations are done with the model and data on the same device
    # in our case, we haven't setup device-agnostic code yet so our data and model are
    # on the CPU by default.
    # model_0.to(device)
    # X_test = X_test.to(device)
    y_preds = model_0(X_test)

print("\ny_preds", y_preds)

# SAVE IT
# torch.save(model_0.state_dict(), "my_model-2.pth")

# plot_predictions(predictions=y_preds)
plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)

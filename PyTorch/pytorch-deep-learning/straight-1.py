import torch
from torch import nn
import matplotlib.pyplot as plt

print("Torch version:", torch.__version__)

weight = 0.7  # b (slope)
bias = 0.3  # a (y intercept)

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))  # 80% of data used for training set, 20% for testing
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show()


plot_predictions()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)

model_0 = LinearRegressionModel()

list(model_0.parameters())

model_0.state_dict()

with torch.inference_mode():
    y_preds = model_0(X_test)

print(f"Number of testing samples: {len(X_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

plot_predictions(predictions=y_preds)

# TODO: Straight #1, here we go.

# y_test - y_preds

loss_fn = nn.L1Loss()  # MAE loss is same as L1Loss

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

torch.manual_seed(42)

epochs = 100

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

        # 2. Caculate loss on test data
        test_loss = loss_fn(test_pred, y_test.type(
            torch.float))  # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

        # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)

            train_loss_values.append(loss.detach().numpy())

            test_loss_values.append(test_loss.detach().numpy())

            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

plt.plot(epoch_count, train_loss_values, label="Train loss")

plt.plot(epoch_count, test_loss_values, label="Test loss")

plt.title("Training and test loss curves")

plt.ylabel("Loss")

plt.xlabel("Epochs")

plt.legend()

print("The model learned the following values for weights and bias:")
print(model_0.state_dict())

print("\nAnd the original values for weights and bias are:")

print(f"weights: {weight}, bias: {bias}")

model_0.eval()

with torch.inference_mode():
    # 3. Make sure the calculations are done with the model and data on the same device
    # in our case, we haven't setup device-agnostic code yet so our data and model are
    # on the CPU by default.
    # model_0.to(device)
    # X_test = X_test.to(device)

    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)

from pathlib import Path

MODEL_PATH = Path("models")

MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_0.pth"

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")

torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# TODO: What?
# get_ipython().system('ls -l models/01_pytorch_workflow_model_0.pth')

loaded_model_0 = LinearRegressionModel()

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_0.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

y_preds == loaded_model_preds

import torch

from torch import nn

import matplotlib.pyplot as plt

torch.__version__

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

weight = 0.7

bias = 0.3

start = 0

end = 1

step = 0.02

X = torch.arange(start, end, step).unsqueeze(
    dim=1)  # without unsqueeze, errors will happen later on (shapes within linear layers)

y = weight * X + bias

X[:10], y[:10]

train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]

X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

plot_predictions(X_train, y_train, X_test, y_test)


class LinearRegressionModelV2(nn.Module):

    def __init__(self):
        super().__init__()

        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)

model_1 = LinearRegressionModelV2()

model_1, model_1.state_dict()

next(model_1.parameters()).device

model_1.to(device)  # the device variable was set above to be "cuda" if available or "cpu" if not

next(model_1.parameters()).device

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

torch.manual_seed(42)

epochs = 1000

X_train = X_train.to(device)

X_test = X_test.to(device)

y_train = y_train.to(device)

y_test = y_test.to(device)

for epoch in range(epochs):

    # TRAINING

    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    # TESTING

    model_1.eval()  # put the model in evaluation mode for testing (inference)

    # 1. Forward pass
    with torch.inference_mode():

        test_pred = model_1(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

from pprint import pprint  # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html

print("The model learned the following values for weights and bias:")

pprint(model_1.state_dict())

print("\nAnd the original values for weights and bias are:")

print(f"weights: {weight}, bias: {bias}")

model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)

# y_preds

plot_predictions(predictions=y_preds.cpu())

from pathlib import Path

MODEL_PATH = Path("models")

MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_1.pth"

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")

torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_1 = LinearRegressionModelV2()

loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

loaded_model_1.to(device)

print(f"Loaded model:\n{loaded_model_1}")

print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")

loaded_model_1.eval()

with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

y_preds == loaded_model_1_preds

import torch
from torch import nn
import matplotlib.pyplot as plt

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


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    # Training data = blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Test data = green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Predictions = red
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Legend
    plt.legend(prop={"size": 14})

    plt.show()


# Define your model class
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # define your model layers here
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    # Forward = computation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)

# LOAD THE PRE-TRAINED MODEL
PATH = "my_model.pth"
model_0 = MyModel()
model_0.load_state_dict(torch.load(PATH))

"""
TRAINING & TESTING LOOP
"""

# Create loss function
loss_fn = nn.L1Loss()  # L1Loss = Mean absolute error

# Create optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 100

train_loss_values = []

test_loss_values = []

epoch_count = []

for epoch in range(epochs):
    """
    TRAINING / OPTIMIZATION
    """
    model_0.train()  # Put model in TRAINING MODE

    y_pred = model_0(X_train)  # 1. Forward pass on TRAIN data

    loss = loss_fn(y_pred, y_train)  # 2. Calculate the loss

    optimizer.zero_grad()  # 3. Zero grad of the optimizer

    loss.backward()  # 4. Loss backwards

    optimizer.step()  # 5. Progress the optimizer

    # TESTING
    model_0.eval()  # Put model in EVALUATION MODE

    with torch.inference_mode():
        test_pred = model_0(X_test)  # 1. Forward pass on TEST data

        test_loss = loss_fn(test_pred, y_test.type(torch.float))  # 2. Calculate loss on TEST data

        # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# PLOT LOSS CURVES
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

print("\nThe model learned the following values for weights and bias:")
print(model_0.state_dict())

print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

model_0.eval()  # Again?

# MAKE PREDICTIONS
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)

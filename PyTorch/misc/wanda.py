"""
Trains a PyTorch neural network model using stochastic gradient descent optimizer and binary cross-entropy loss,
logs gradients and performance metrics using WandB, and visualizes the model's performance over the training epochs.

pip install wandb
See: wandb.me/intro-pt
"""
import torch
import torch.nn as nn
import wandb

input_shape = 10
train_loader = []
tensor_train_x = []
test_y = []
y_pred = []

# Init new weights & biases run
wandb.init(project="PyTorch Experiments",
           config={"learning_rate": 0.001,
                   "epochs": 100,
                   "batch_size": 128,
                   "dropout": 0.5})

model = nn.Sequential(
    nn.Linear(input_shape, 100),
    nn.ReLU(),
    nn.Dropout(wandb.config["dropout"]),
    nn.Linear(100, 1),
    nn.Sigmoid()
)

num_epochs = wandb.config["epochs"]
learning_rate = wandb.config["learning_rate"]

criterion = nn.BCELoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# Log and visualize gradients
wandb.watch(model, criterion, log="all", log_freq=10)

# Define training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

if epoch % 10 == 0:
    epoch_loss = running_loss / len(tensor_train_x)
    val_accuracy = y_pred.eq(test_y).sum() / test_y

    wandb.log({"train_loss": epoch_loss,
               "val_accuracy": val_accuracy},
              step=epoch)

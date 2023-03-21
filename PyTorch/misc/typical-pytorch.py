import torch
import torch.nn as nn
import torch.optim as optim


def load_and_preprocess_data():
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
    return X_train, y_train, X_test, y_test


# Define your hyperparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 32

# Load and preprocess your data
train_data, train_labels, test_data, test_labels = load_and_preprocess_data()


model = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
)

# Define your loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train your model
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        # Get a batch of data and labels
        data_batch = train_data[i:i + batch_size]
        labels_batch = train_labels[i:i + batch_size]

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data_batch)

        # Compute the loss
        loss = loss_fn(outputs, labels_batch)

        # Backward pass and update weights
        loss.backward()
        optimizer.step()

# Evaluate your model on test data
with torch.no_grad():
    outputs = model(test_data)
    test_loss = loss_fn(outputs, test_labels)
    test_acc = (outputs.argmax(dim=1) == test_labels).float().mean()

# Make predictions with your model
with torch.no_grad():
    predictions = model(test_data)

# Print out some results
print('Test accuracy:', test_acc.item())
print('Predictions:', predictions)

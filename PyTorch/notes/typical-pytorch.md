## Typical PyTorch program

```py
import torch
import torch.nn as nn
import torch.optim as optim


def load_and_preprocess_data():
    weight = 0.7
    bias = 0.3

    start = 0
    end = 1
    step = 0.02

    X = torch.arange(start, end, step).unsqueeze(dim=1)  # Returns a 1-D tensor with values from "start" to "end" with "step"
    y = weight * X + bias

    pos = int(0.8 * len(X))
    X_train, y_train = X[:pos], y[:pos]  # Get all elements up until position
    X_test, y_test = X[pos:], y[pos:]  # Get all elements from the position, onwards.

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
        data_batch = train_data[i:i + batch_size]  # From i to i + batch size
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
    test_acc = (outputs.argmax(dim=1) == test_labels).float().mean()  # Accuracy formula

# Make predictions with your model
with torch.no_grad():
    predictions = model(test_data)

# Print out some results
print('Test accuracy:', test_acc.item())
print('Predictions:', predictions)

```

<br>

1. Import the necessary libraries, then define some **hyperparameters:**
    * Learning rate
    * Number of epochs
    * Batch size

2. **Load** and preprocess data.

3. Define our model **architecture** using PyTorch's `nn.Sequential` API.

4. Define our **loss function** and **optimizer** and set the model's parameters to be optimized.

5. **Train** our model using a loop that goes through each epoch and each batch of data.
    6. We first zero out the gradients
    7. Perform a forward pass through the model
    8. Compute the loss
    9. Perform a backward pass
    10. Update the model's weights

6. **Evaluate** our model on test data using the `with torch.no_grad()` context manager to turn off gradient computation. We also make some predictions with the model.

7. **Print out some results**:
    * Test accuracy
    * Predictions

## MNIST classification

*Because Buddy omitted the load\_and\_preprocess\_data() code...*

For what dataset might you define your PyTorch model architecture like the following?

```python
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

<br>

This architecture seems to be suitable for a **classification** task on the MNIST dataset, which is a collection of **28x28** grayscale images of handwritten digits **(0 to 9).**

```c
28 * 28 = 784
```

The input size of **784** corresponds to flattening the 28x28 images into a 1D tensor of length 784.

The output size of **10** represents the number of classes (digits) in the dataset.

The **ReLU** activation function is commonly used in neural networks for classification tasks.

The choice of **64** hidden units in the middle layer is a common choice for a simple and effective architecture.

<br>

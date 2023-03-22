import torch
import torch.nn as nn
from sklearn.datasets import make_multilabel_classification

# Generate a random multilabel classification dataset with 100 samples and 10 features
X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=5)

# Convert the numpy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


# Define a simple neural network with one hidden layer
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# Instantiate the model with the appropriate input, hidden, and output sizes
net = Net(input_size=10, hidden_size=5, output_size=5)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# Train the model for 1000 epochs
for epoch in range(1000):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = net(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print statistics every 100 epochs
    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 1000, loss.item()))

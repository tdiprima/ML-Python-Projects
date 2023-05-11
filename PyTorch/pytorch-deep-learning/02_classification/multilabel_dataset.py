"""
Generate a random multilabel classification dataset with 100 samples and 10 features
"""
import torch
import torch.nn as nn
from sklearn.datasets import make_multilabel_classification
from torchinfo import summary

X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=5)
# print(len(X), len(y))  # 100 100

# Convert the numpy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class Net(nn.Module):
    """
    Define a simple neural network with one hidden layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("\nInput size:", x.size())
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# Instantiate the model with the appropriate input, hidden, and output sizes
net = Net(input_size=10, hidden_size=5, output_size=5)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Notice - no "with logits"
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(1000):
    # Input size: torch.Size([100, 10])
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

# PREDICTIONS
from sklearn.model_selection import train_test_split

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(len(X_train), len(X_test), len(y_train), len(y_test))  # 80 20 80 20

# Make predictions with model
net.eval()
with torch.inference_mode():
    # Input size: torch.Size([20, 10])
    # Forward pass on test data
    test_pred = net(X_test)

    # Calculate loss on test data
    test_loss = criterion(test_pred, y_test.type(torch.float))

# Check the predictions
print(f"\nNumber of testing samples: {len(X_test)}")
print(f"Number of predictions made: {len(test_pred)}")
# print(f"\nPredicted values:\n{test_pred}")
print("\nHow close were we? (test_loss)", test_loss)

print()
summary(net, input_size=[100, 10])

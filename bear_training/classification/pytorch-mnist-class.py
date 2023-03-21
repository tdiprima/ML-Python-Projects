"""
Works, but run on server if increase hyper-parameters
Multi-class classification
"""
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define your hyper-parameters
num_epochs = 10
# batch_size = 32
batch_size = 64

# Define the transform to preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image data
])

# Download and load the MNIST dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Length: 938
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Length: 157

# Define the data loader to iterate over the dataset in batches
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


# Define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = torch.nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


# Instantiate the neural network
net = Net()

# Define cross-entropy loss function
criterion = torch.nn.CrossEntropyLoss()

# Define Adam optimizer
optimizer = torch.optim.Adam(net.parameters())

# Train the neural network
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Print the losses every n epochs
        if i % 100 == 99:  # Means 99, 199, 299, etc.
            print(f'Epoch {epoch + 1}, batch {i + 1}: loss {running_loss / 100:.3f}')
            running_loss = 0.0

# Evaluate the neural network on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test accuracy: {accuracy:.2f}%')

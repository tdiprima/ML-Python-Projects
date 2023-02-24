"""
Doesn't work.
PyTorch UNet Image Segmentation.md
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# Define the UNet architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Define the encoder layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        # Define the decoder layers
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        # Decoder
        x = self.upconv1(x2)

        try:
            x = torch.cat([x, x1], dim=1)
        except Exception as ex:
            print("torch.cat:", ex)
            exit(1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.upconv2(x)
        x = self.conv4(x)

        return x


# Define the training function
def train(model, device, train_loader, optimizer, criterion):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        try:
            output = model(data)
        except Exception as es:
            print("output:", es)
            exit(1)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# Define the validation function
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader.dataset)
    print('Validation Loss: {:.4f}'.format(val_loss))


# Define the main function
def main():
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    transform = transforms.Compose([
        transforms.ToTensor(),

    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Set up the model, optimizer, and loss function
    model = UNet(in_channels=3, out_channels=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, criterion)
        validate(model, device, val_loader, criterion)

    # Display some sample predictions
    sample_batch = next(iter(val_loader))
    sample_data, _ = sample_batch
    sample_data = sample_data.to(device)

    model.eval()
    with torch.no_grad():
        sample_output = model(sample_data)
    sample_output = torch.argmax(sample_output, dim=1)

    try:
        torchvision.utils.save_image(sample_data.cpu(), 'sample_data.png')
        torchvision.utils.save_image(sample_output.cpu().unsqueeze(1), 'sample_output.png')
    except Exception as ex:
        print("torchvision:", ex)
        exit(1)


if __name__ == '__main__':
    main()

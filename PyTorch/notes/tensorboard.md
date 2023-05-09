## Quick lesson

Just look at the rest of the [code](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md#a-simple-convnet-to-start-with).

```sh
tensorboard --logdir=runs
```

Running locally: http://localhost:6006/

[how-to-use-tensorboard-with-pytorch.md](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md)

```py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST

class ConvNet(nn.Module):
    """
    Simple Convolutional Neural Network for classifying MNIST digits
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 5, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 24 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        """
        Forward pass
        """
        return self.layers(x)

if __name__ == '__main__':
    # Initialize the SummaryWriter for TensorBoard
    # Its output will be written to ./runs/
    writer = SummaryWriter()

    # Initialize the ConvNet
    convnet = ConvNet()

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(convnet.parameters(), lr=1e-4)  # 0.0001

    # La la la...

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum
        # Lulu...
        for i, data in enumerate(trainloader, 0):
            # Deedee...

            # Perform forward pass
            outputs = convnet(inputs)
```

<br>

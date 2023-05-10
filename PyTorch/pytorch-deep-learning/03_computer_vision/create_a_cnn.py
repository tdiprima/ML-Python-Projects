import torch
from torch import nn
from torchinfo import summary


class FashionMNISTModel(nn.Module):
    """
    Model architecture that replicates the TinyVGG
    model from the CNN Explainer website
    """

    # Initialize the class
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        # Initialize the initializer
        super().__init__()
        """
        Create NN in a couple of blocks
        Create Layers inside nn.Sequential
        First 2 layers are feature extractors (learning patterns that best represent our data).
        Last layer (output layer) is classifier layer (classify features into our target classes).
        """
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.classifier = nn.Sequential(nn.Flatten(),  # Flatten to single feature vector
                                        nn.Linear(in_features=hidden_units * 7 * 7,  # in_features=hidden_units * 0,
                                                  out_features=output_shape
                                                  # The length of how many classes we have. One value for each class.
                                                  ))

    def forward(self, x):
        print('\nInput size:', x.size())
        x = self.conv_block_1(x)
        print(1, "x.shape:", x.shape)
        x = self.conv_block_2(x)
        print(2, "x.shape:", x.shape)
        x = self.classifier(x)
        print(3, "x.shape:", x.shape)
        return x


model = FashionMNISTModel(input_shape=1, hidden_units=10, output_shape=10)

# create dummy tensor
input_tensor = torch.randn(1, 1, 28, 28)

# pass through model and check output shape
output_tensor = model(input_tensor)
assert output_tensor.shape == (1, 10)
print(f"\noutput_tensor shape: {output_tensor.shape}")

summary(model, input_size=[1, 1, 28, 28])

"""
1 x.shape: torch.Size([1, 10, 14, 14])
2 x.shape: torch.Size([1, 10, 7, 7])
3 x.shape: torch.Size([1, 10])
torch.Size([1, 10])
"""

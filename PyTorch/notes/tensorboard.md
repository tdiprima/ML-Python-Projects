## Quick lesson

```py
import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# Train a simple model and create TensorBoard logs
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


model = create_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

hparams_callback = hp.KerasCallback(log_dir, {
    'num_relu_units': 512,
    'dropout': 0.2
})

"""
TensorBoard logs are created during training by passing the TensorBoard and hyperparameters
callbacks to Keras' Model.fit(). These logs can then be uploaded to TensorBoard.dev.
"""
model.fit(
    x=x_train,
    y=y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, hparams_callback])
```

## "Convoluted" is right!

Just look at the rest of the [code](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md#a-simple-convnet-to-start-with).

```sh
tensorboard --logdir=runs
```

Running locally: http://localhost:6006/

[how-to-use-tensorboard-with-pytorch.md ](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md)

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
    optimizer = torch.optim.Adam(convnet.parameters(), lr=1e-4)

    # La la la

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum
        # Lulu!
        for i, data in enumerate(trainloader, 0):
            # Deedee

            # Perform forward pass
            outputs = convnet(inputs)
```

<br>

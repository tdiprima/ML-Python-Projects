## CNN Architecture: 13x13 Mystery

Question: Why 13?

```py
class TinyVGG(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(...)

        self.conv_block_2 = nn.Sequential(...)

        # RIGHT HERE:
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
```

See: <a href="../pytorch-deep-learning/04_custom_datasets/test_train_def.py">test\_train\_def.py</a>, <a href="../pytorch-deep-learning/03_computer_vision/create_a_cnn.py">create\_a\_cnn.py</a>.

Alright, so you're asking about the `13 * 13` part in `nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape)`. To understand why the number 13 is used, we need to consider how the Convolutional Neural Network (CNN) processes an image through its layers.

Let's break it down:

1. **Convolution Layers**: The purpose of these layers is to learn and identify features from the input image, like edges, corners, textures, etc.
2. **Pooling Layers**: These are used to reduce the dimensions of the image. This makes computations faster and less memory-intensive, while still keeping the important features.

In your `TinyVGG` model, each convolutional block consists of two convolutional layers (`nn.Conv2d`) followed by a pooling layer (`nn.MaxPool2d`).

### Now let's dig into how the number 13 is calculated.

```py
self.conv_block_1 = nn.Sequential(
    nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)  # default stride value is same as kernel_size
)
```

Assume your input image is square, let's say `32x32` pixels. After the **first convolution** with `kernel_size=3` and `stride=1`, the dimensions would be:

- (32 - 3) + 1 = 30
- (32 - 3) + 1 = 30

So the output of the first convolutional layer would be `30x30`.

Then comes the **second convolution** with the same kernel size and stride:

- (30 - 3) + 1 = 28
- (30 - 3) + 1 = 28

So the output after the second convolutional layer would be `28x28`.

Then you do **max-pooling** with `kernel_size=2` and `stride=2`. This reduces the dimensions by half:

- 28 / 2 = 14
- 28 / 2 = 14

So after the first `conv_block_1`, the dimensions would be `14x14`.

**Then comes `conv_block_2`**, which performs the same operations. After its convolutions and max-pooling, the dimensions would be:

- (14 - 3) + 1 = 12
- (12 - 3) + 1 = 10
- 10 / 2 = 5

Notice that the dimensions are reduced to `5x5`.  Therefore, the `nn.Linear` layer should have `in_features=hidden_units * 5 * 5` instead of `hidden_units * 13 * 13`.

## Actual Numbers

**Input size:** `torch.Size([32, 3, 64, 64])`

* Batch size 32
* Channels 3
* 64x64

Since it's 64x64:

```js
var x = 64;

var res = (x - 3) + 1;

newRes = (res - 3) + 1;

var conv_block_1 = newRes / 2;

console.log("conv_block_1", conv_block_1);


res = (conv_block_1 - 3) + 1;
newRes = (res - 3) + 1;
var conv_block_2 = newRes / 2;

console.log("conv_block_2", conv_block_2);

// conv_block_1 30
// conv_block_2 13
```

And that's how we get 13.

<br>

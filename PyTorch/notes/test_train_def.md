## CNN architecture design

If the model's input size is [32, 3, 64, 64], and we define the model class's init function as follows, how do we know to flatten it with `hidden_units * 13 * 13`?  Why 13?

See: test_train_def.py

Also: create_a_cnn.py

<mark>**Answer!**</mark>  Behold the `forward` method:

```
Input size: torch.Size([32, 3, 64, 64])
1 x.shape: torch.Size([32, 10, 30, 30])
2 x.shape: torch.Size([32, 10, 13, 13])
3 x.shape: torch.Size([32, 3])
```

It's halved, then minus 2.

That's all you need to do, is print it out.

```python
self.classifier(self.conv_block_2(self.conv_block_1(x)))
# becomes
x = self.conv_block_1(x)
x = self.conv_block_2(x)
x = self.classifier(x)
# you just print in between, and add this at the top of forward:
print("Input size:", x.size())
```

Follow along with model summary:

```python
from torchinfo import summary
model = TinyVGG(...)
summary(model, input_size=[32, 3, 64, 64])
```

<br>

The flattening size `hidden_units * 13 * 13` is determined by the architecture of the model. The input size is `[32, 3, 64, 64]`, meaning it has 32 images in the batch, with each image having 3 color channels and a size of 64x64 pixels. 

After passing through the two convolutional blocks and max pooling layers, the spatial dimensions of the output tensor are **reduced by half at each max pooling layer.** The first max pooling layer reduces the spatial dimensions from 64x64 to 32x32 (then -2=30), and the second max pooling layer reduces them further to 15x15 (-2=13).

This (halving) is because the max pooling layer has a **kernel size of 2 and a stride of 2**, which reduces the size of the tensor by a factor of 2 in each spatial dimension.

Minus 2, because kernel size is 3 and stride is 1.

Finally, the output of the second convolutional block is **flattened into a 1D tensor** using the `nn.Flatten()` layer, which concatenates all the spatial dimensions of the tensor into a single dimension.

At this point, the tensor has a shape of `[batch_size, hidden_units * 13 * 13]`.

<br>

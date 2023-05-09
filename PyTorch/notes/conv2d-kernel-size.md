## Conv2d kernel_size

<span style="color:#0000dd;">When creating a pytorch nn.Conv2d, in what situations would we set kernel_size, and how do we know what size to set it to?</span>

When creating a `nn.Conv2d` layer in PyTorch, the `kernel_size` parameter specifies the size of the kernel (also known as filter or window) that slides over the input data to perform the convolution operation. The kernel size is typically a square matrix, and its dimensions are specified as a tuple, e.g., `kernel_size=(3, 3)`.

You would set the `kernel_size` based on the characteristics of your input data and the nature of the problem you are trying to solve. Generally, smaller kernel sizes allow for more local features to be learned, while larger kernel sizes can capture more global features. In practice, the choice of kernel size is often determined empirically through experimentation and tuning.

For example, in image classification tasks, it is common to start with a small kernel size, such as `kernel_size=(3, 3)`, and gradually increase it to capture more global features as the network goes deeper. In contrast, in object detection tasks, where the objects of interest can vary in size, a range of kernel sizes may be used to capture features at different scales.

Overall, the choice of kernel size is dependent on the nature of the problem, the complexity of the data, and the design of the network. It is a hyperparameter that needs to be carefully tuned to achieve optimal performance.

<span style="color:#0000dd;">What if it's `kernel_size=3`?  Is that shorthand for `(3, 3)`?</span>

Yes, when the `kernel_size` parameter is set to an integer, such as `kernel_size=3`, it is interpreted as a square kernel of size `(3, 3)`. In other words, `kernel_size=3` is a shorthand for `kernel_size=(3, 3)`. 

Similarly, if you set `kernel_size=5`, it would be interpreted as a square kernel of size `(5, 5)`. This shorthand notation can be used when you want to use a square kernel with equal dimensions in both height and width, and you don't want to type out the tuple explicitly.

It's worth noting that, in practice, using square kernels is not always necessary or optimal. Depending on the task and the input data, rectangular kernels (i.e., kernels with different dimensions for height and width) may be more appropriate. In such cases, you would need to specify the `kernel_size` as a tuple, rather than using the shorthand notation.

## Why are we multiplying by 26 squared?

```py
self.fc1 = nn.Linear(hidden_units * 26 * 26, output_shape)
```

The reason why you are multiplying by 26x26 in the input shape of the `Linear` layer is **because of the way convolutional layers work.**

When a convolutional layer is applied to an input image, **it slides a small window** (the **kernel**) over the image, and computes **dot products** between the kernel and the corresponding pixel values in the input.

The size of the output feature map is determined by the size of the **input** image, the size of the **kernel**, and the **stride** (how much the kernel is moved between each computation).

In your case, you haven't specified the details of the convolutional layers that come before the `Linear` layer, but assuming that the input image size is indeed 28x28, it's likely that the convolutional layers are using kernels with size 3x3 and a stride of 1.

```py
# Mm-hm
self.conv1 = nn.Conv2d(input_shape[0], hidden_units, kernel_size=3
```

This means that each convolutional layer will **reduce** the size of the input feature map by **2 pixels** in each dimension (due to the kernel size and stride). Therefore, after two convolutional layers, the output feature map size would be 26x26.

So, when you specify the input shape of the `Linear` layer as `hidden_units * 26 * 26`, you are **flattening the output** feature map from the previous convolutional layer(s) **into a 1D vector,** where each element corresponds to a feature map activation.

The `hidden_units` parameter specifies the number of neurons in the `Linear` layer, which will be fully connected to the flattened input feature map. 

Therefore, the **input** shape of the `Linear` layer is determined by the **number of neurons** in the `Linear` layer and the size of the **flattened** feature map, which is `hidden_units * 26 * 26`.

## Flatten

<span style="color:#0000dd;">Do we always perform a flattening step in our models before the forward pass?</span>

No, the need for flattening depends on the structure of the model architecture and the shape of the input data.

In convolutional neural networks (CNNs), the **input** data is typically **4-dimensional**, with dimensions representing the batch size, number of channels, height, and width of the input image.

The **output** of the convolutional layers is also a **4-dimensional** tensor, but with different dimensions depending on the number of filters used and the size of the input.

When using **fully connected layers** (also known as **linear layers**) in a CNN, it is often necessary to **flatten** the output of the convolutional layers into a **2-dimensional** tensor (batch size, flattened feature map) before passing it to the fully connected layers.

This is because fully connected layers require a 2D tensor as input, where the first dimension is the batch size and the second dimension is the flattened feature map.

However, some CNN architectures, such as the Inception family of models, use other types of layers (such as 1x1 convolutions) **to avoid flattening** the feature maps. In these cases, the output of the convolutional layers can be passed directly to the subsequent layers without flattening.

On the other hand, in other types of neural network architectures, such as recurrent neural networks (RNNs), the input data can be a sequence of variable length, and **flattening is not necessary.** Instead, the input sequence is processed by the network one element at a time, and the output of each step is passed to the next step until the entire sequence has been processed.

Therefore, the need for flattening depends on the structure of the network and the shape of the input data, and it is not always required in all types of networks.

<br>

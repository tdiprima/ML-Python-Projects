## Create 3D Tensor

The `torch.randn(2, 3, 5)` function call creates a tensor with shape `(2, 3, 5)`.

This means that the tensor has 3 dimensions, with the first dimension having a length of 2, the second dimension having a length of 3, and the third dimension having a length of 5.

So the resulting tensor would have 2 "layers", each with 3 rows and 5 columns.

You can think of it as a 3D matrix with 2 "slices", each slice having 3 rows and 5 columns.

```python
muerte = [[[-0.5389, 0.4626, -0.8719, -0.0271, -0.3532],
           [1.4639, 0.1729, 1.0514, 0.8539, 0.5130],
           [0.5397, 0.5655, 0.5058, 0.2225, -1.3793]],

          [[0.6258, -2.5850, -0.0240, -0.1222, -0.7470],
           [1.7093, 0.0579, 1.1930, 1.9373, 0.7287],
           [0.9809, 0.4146, 1.1566, 0.2691, -0.0366]]]
```

<br>

To create a matrix with 2 rows and 3 columns, you would use `torch.randn(2, 3)`.

This would create a 2D tensor with 2 rows and 3 columns.

```python
salvavidas = [[-0.7581, 0.9989, -0.8793],
              [0.7486, -1.3375, 0.6449]]
```

## Permute

Swaps the order of the dimensions.

The 2nd dimension is first, the first dim is in the middle, and the 1st dimension is last.
So all we did was take the last and make it first.  Cute.

```python
torch.permute(x, (2, 0, 1))
# so the new shape is 5, 2, 3
```

## Match Tensor Sizes

```
torch.cat: Sizes of tensors must match except in dimension 1.
Expected size 64 but got size 32 for tensor number 1 in the list.
```

<br>
<span style="color: #000088; font-size:larger">I have one torch tensor of shape [64, 64, 64, 64], and another torch tensor of shape [64, 64, 32, 32].  How do I make the sizes of tensors match?</span>

To make the sizes of the two tensors match, you can use the `torch.nn.functional.interpolate()` function to upsample or downsample the second tensor to the same shape as the first tensor. Here's an example code snippet that upsamples the second tensor to the shape of the first tensor:

```py
import torch.nn.functional as F

# Assume the two tensors are named tensor1 and tensor2
new_tensor2 = F.interpolate(tensor2, size=(64, 64, 64), mode='trilinear', align_corners=False)

# The new_tensor2 will have shape [64, 64, 64, 64]
```

<br>

In the above code, `F.interpolate()` function takes the following parameters:

* `tensor2`: The tensor to be resized
* `size`: The new size of the tensor, which is specified as a tuple of dimensions
* `mode`: The interpolation mode. 'trilinear' is used for 3D data, which performs linear interpolation in 3 dimensions
* `align_corners`: Specifies whether the corners of the input and output tensors should be aligned. In this case, we set it to False.

After calling `F.interpolate()`, the new tensor will have the same shape as the first tensor, i.e., [64, 64, 64, 64].

<br>

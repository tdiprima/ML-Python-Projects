## Modifying PyTorch Tensor

Let's break it down step by step.

1. Create a one-dimensional tensor `x`.
   * `torch.arange(1., 10.)`
   * Basic array containing values from 1.0 to 9.0.

1. Change the view of `x` to a two-dimensional tensor `z`.
   * `z = x.view(1, 9)`
   * One row, nine columns.
   * Or, we could've done: `x_reshaped = x.reshape(1, 9)`.

```javascript
blah = [
    [1., 2., 3., 4., 5., 6., 7., 8., 9.]
]
```

* When you index into `z` using `z[:, 0]`, you are selecting all rows (`:`) and the first column (`0`) of `z`. This returns a one-dimensional tensor of length one, which contains the value 1.0.

* When you assign the value 5 to `z[:, 0]` using `z[:, 0] = 5`, you are modifying the first element of the first row of `z`, which now becomes 5. This means that the first element of `x` also becomes 5, since `z` and `x` share the same underlying data.

So in summary, the operation `z[:, 0] = 5` sets the first element of the first row of the two-dimensional tensor `z` to 5, and this change is also reflected in the original one-dimensional tensor `x`.

## What does indexing into a tensor mean?

Indexing into a tensor means accessing its individual elements or a subset of its elements by specifying their position(s) in the tensor.

In PyTorch, you can index into a tensor using square brackets (`[]`). The index or indices that you provide specify the position(s) of the element(s) you want to access. The indexing syntax can take various forms depending on the number of dimensions in the tensor and the type of indexing you want to perform.

Here are a few examples:

- **Indexing a one-dimensional tensor**: If you have a one-dimensional tensor `x`, you can access its elements using an integer index between 0 and `len(x) - 1`. For example, to access the first element of `x`, you can use `x[0]`.

- **Indexing a two-dimensional tensor**: If you have a two-dimensional tensor `x`, you can access its elements using two indices, one for the row and one for the column. For example, to access the element in the second row and third column of `x`, you can use `x[1, 2]`.

- **Slicing a tensor**: You can also use slicing to access a subset of the elements in a tensor. Slicing is done using the colon (`:`) operator to specify the range of indices you want to include. For example, to access the first three elements of a one-dimensional tensor `x`, you can use `x[:3]`. To access the first two rows and the first three columns of a two-dimensional tensor `y`, you can use `y[:2, :3]`.

Keep in mind that indexing into a tensor returns a new tensor that shares the same underlying data as the original tensor. So modifying the elements of the new tensor will also modify the elements of the original tensor.

## Explicar

```python
# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0)

# Comes out like this:
enloquecido = [
	[1., 2., 3., 4., 5., 6., 7., 8., 9.],
	[1., 2., 3., 4., 5., 6., 7., 8., 9.],
	[1., 2., 3., 4., 5., 6., 7., 8., 9.],
	[1., 2., 3., 4., 5., 6., 7., 8., 9.]
]

# 1st dimension
x_stacked = torch.stack([x, x, x, x], dim=1)

pantera_fuerte = [
    [1., 1., 1., 1.],
    [2., 2., 2., 2.],
    [3., 3., 3., 3.],
    [4., 4., 4., 4.],
    [5., 5., 5., 5.],
    [6., 6., 6., 6.],
    [7., 7., 7., 7.],
    [8., 8., 8., 8.],
    [9., 9., 9., 9.]
]
```

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

To create a matrix with 2 rows and 3 columns, you would use `torch.randn(2, 3)`.

This would create a 2D tensor with 2 rows and 3 columns.

```python
salvavidas = [[-0.7581, 0.9989, -0.8793],
              [0.7486, -1.3375, 0.6449]]
```

## Permute

Swaps the order of the dimensions.

Think of permute as "Who's On First?" ⚾️

The 2nd dimension is first, the first dim is in the middle, and the 1st dimension is last.
So all we did was take the last and make it first.  Cute.

```python
torch.permute(x, (2, 0, 1))
# so the new shape is 5, 2, 3
```
## What is a tensor?

In the context of machine learning and deep learning, tensors are used to represent the **input and output data**, as well as the **parameters** and intermediate **activations** of neural network models.

In a neural network, a tensor is typically represented as a multi-dimensional **array of numbers**, with each dimension corresponding to a **different feature or aspect** of the data.

In **PyTorch**, tensors are the fundamental building blocks of neural networks and machine learning models, and the framework provides a wide range of **operations and functions** for working with tensors efficiently and effectively.

## Visual Representation

Let's say we have a 3-dimensional tensor that represents a collection of images, each with dimensions of 32 pixels by 32 pixels and 3 color channels.
(Well, already that's 4-D, or 4 axes...)
We can represent this tensor as a block of data with N axes:

```
+------------------------------------------+
|                  Tensor                  |
+------------------------------------------+
|         | Width | Height | Channels      |
|---------|-------|--------|--------------|
| Image 1 | 32    | 32     | 3            |
| Image 2 | 32    | 32     | 3            |
| ...     | ...   | ...    | ...          |
| Image N | 32    | 32     | 3            |
+------------------------------------------+
```

1. The first axis of the tensor represents the individual images
2. The second axis represents the width of each image
3. The third axis represents the height
4. Finally, the color channels of each image.

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

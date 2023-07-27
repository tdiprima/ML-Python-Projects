## Modifying a PyTorch Tensor

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

<br>

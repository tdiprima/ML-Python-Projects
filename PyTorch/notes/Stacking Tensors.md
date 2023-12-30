## Stacking Tensors in PyTorch

<span style="color:blue;font-size:larger;">Is PyTorch torch.vstack the same as doing torch.stack dim=0?</span>

```py
x = torch.arange(1., 10.)
torch.stack([x, x, x, x], dim=0)
```

No, `torch.vstack` and `torch.stack([x, x, x, x], dim=0)` are not the same in PyTorch when used with the tensor `x = torch.arange(1., 10.)`. Let's understand the difference:

1. **`torch.vstack`**: This function is used for stacking tensors in sequence vertically (row-wise). In the context of 1D tensors, `torch.vstack` will treat each tensor as a row and stack them one below the other, effectively turning them into a 2D tensor where each original tensor becomes a row.

2. **`torch.stack([x, x, x, x], dim=0)`**: This function stacks a sequence of tensors along a new dimension (at the specified dimension index). When `dim=0`, it stacks the tensors along the first dimension. In the case of stacking 1D tensors, this will result in a 2D tensor where each original tensor is a slice along the first dimension.

The key difference is how the dimensions are treated. `torch.vstack` turns 1D tensors into rows of a 2D tensor, while `torch.stack` with `dim=0` creates a new dimension and stacks the 1D tensors along it.

Let's illustrate this with the tensor `x = torch.arange(1., 10.)` to see the difference in results.

The results from `torch.vstack([x, x, x, x])` and `torch.stack([x, x, x, x], dim=0)` with the tensor `x = torch.arange(1., 10.)` are:

- Result from `torch.vstack`:

  ```c
  tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.],
          [1., 2., 3., 4., 5., 6., 7., 8., 9.],
          [1., 2., 3., 4., 5., 6., 7., 8., 9.],
          [1., 2., 3., 4., 5., 6., 7., 8., 9.]])
  ```

- Result from `torch.stack([x, x, x, x], dim=0)`:

  ```c
  tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.],
          [1., 2., 3., 4., 5., 6., 7., 8., 9.],
          [1., 2., 3., 4., 5., 6., 7., 8., 9.],
          [1., 2., 3., 4., 5., 6., 7., 8., 9.]])
  ```

In this specific case, since `x` is a 1D tensor, both `torch.vstack` and `torch.stack` with `dim=0` produce the same 2D tensor, where each row is a copy of the tensor `x`. However, the conceptual difference remains: `torch.vstack` is for vertical stacking, treating each tensor as a row, while `torch.stack` with `dim=0` creates a new dimension at the beginning. In more complex cases with higher-dimensional tensors, the difference would be more pronounced.

```py
import torch

# Creating a tensor x
x = torch.arange(1., 10.)

# Using torch.vstack
vstack_result = torch.vstack([x, x, x, x])

# Using torch.stack with dim=0
stack_result = torch.stack([x, x, x, x], dim=0)

vstack_result, stack_result
```

<br>

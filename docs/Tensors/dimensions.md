## Dimensions

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*zcidDaCCmJeD8y-9.png" width="600">

<!-- https://hadrienj.github.io/deep-learning-book-series-home/ -->

The way to remember how many dimensions a tensor has, is by **counting the number of square brackets.**

### Vector

Described by a torch tensor.

```ruby
vector = torch.tensor([7, 7])
vector.ndim   # 1 set of brackets
vector.shape  # 2 (it's 2 by 1 elements)
```

### Matrix

```ruby
MATRIX = torch.tensor([[7, 8], [8, 10]])
MATRIX.ndim   # 2 (count the number of square brackets on the outside of one side)
MATRIX.shape  # [2, 2] two elements deep, two elements wide
```

<br>
A tensor is essentially a multi-dimensional array.

Here is an HTML table that represents the 2D tensor.

<table border="1">
  <tr>
    <td>7</td>
    <td align="right">8</td>
  </tr>
  <tr>
    <td>8</td>
    <td>10</td>
  </tr>
</table>

<br>
Each cell in the table represents an element in the tensor.

The 1st row of the table `[7, 8]` corresponds to the first row of the tensor.

The 2nd row of the table `[8, 10]` corresponds to the second row of the tensor.

<br>

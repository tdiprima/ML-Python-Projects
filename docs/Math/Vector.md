## What is a Vector

If you're driving a car, your velocity is a vector quantity because it has both a 

* magnitude: how fast you're going
* direction: which way you're heading

### Car Velocity

"The car is moving with a velocity of 60 kilometers per hour to the east."

This sentence indicates both the magnitude (60 kilometers per hour) and the direction (to the east) of the car's velocity, which are the two components necessary to fully describe velocity as a vector quantity.

### Force vector

Here's how you might describe a force vector in a sentence:

"The object experiences a force of 50 newtons directed upwards."

This sentence tells us the magnitude (50 newtons) and the direction (upwards) of the force, which are the two components necessary to fully describe force as a vector quantity.

<img src="https://calcworkshop.com/wp-content/uploads/ramp-force-vector.png" width="600">

## Dimensions

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*zcidDaCCmJeD8y-9.png" width="600">

<!-- https://hadrienj.github.io/deep-learning-book-series-home/ -->

The way to remember how many dimensions a tensor has, is by **counting the number of square brackets.**

### Vector

Described by a torch tensor.

```py
vector = torch.tensor([7, 7])
vector.ndim   # 1 set of brackets
vector.shape  # 2 (it's 2 by 1 elements)
```

### Matrix

```py
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
Each cell in the table represents an element in the tensor, and each row of the table represents a row in the tensor.

The first row of the table `[7, 8]` corresponds to the first row of the tensor, and the second row of the table `[8, 10]` corresponds to the second row of the tensor.

<br>

<hr>

<img src="https://thirdspacelearning.com/wp-content/uploads/2021/11/Magnitude-of-a-Vector-what-is.png" width="600">

<br>

## array.reshape()

What does it mean, using numpy, when someone does `array.reshape(-1, 1)`?

<span style="color:#0000dd;font-size:larger;">-1 tells the tensor to calculate the size of the dimension in that place.</span>

In NumPy, the `reshape()` function is used to change the shape (dimensionality) of an array. The argument -1 in `array.reshape(-1, 1)` is a placeholder for an unknown dimension size, which is inferred by NumPy based on the total number of elements in the array.

When -1 is passed as the first argument to `reshape()`, NumPy automatically calculates the number of rows required for the reshaped array based on the size of the second argument, which in this case is 1. Therefore, `array.reshape(-1, 1)` reshapes the input array to a column vector (i.e., a 2D array with a single column) while keeping the original number of rows.

For example, consider the following code:

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # shape (3, 3)
b = a.reshape(-1, 1)  # shape it to "idk" rows and 1 column

print(a)
print(b)  # shape (9, 1)
```

The output would be:

```css
[[1 2 3]
 [4 5 6]
 [7 8 9]]

[[1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]]
```

Here, the input array a has 3 rows and 3 columns, and `a.reshape(-1, 1)` reshapes it to a column vector b with 9 rows and 1 column, while keeping the original number of rows.

### See also:

* "TensorFlow/TensorFlow.js/Model Input Shape Mismatch.md"
* PyTorch/notes/dummy.md
* docs/Display.md

<br>

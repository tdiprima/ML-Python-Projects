## Element-Wise Operations

<!-- normalization -->
<span style="color:#0000dd;">What does the dot mean at the end?</span>

```py
# Normalize pixel palues
train_set_x = train_set_x_flatten/255.
```

<br>
It's to indicate that the division operation should be performed element-wise on the array.

In this case, `train_set_x_flatten` is likely a numpy array containing pixel values of grayscale images.

By dividing each element of the array by 255, the pixel values are normalized to be between 0 and 1, which can be beneficial for training certain types of machine learning models.

Without the dot at the end, the division operation would not be performed element-wise, and an error would occur since dividing an array by a scalar requires broadcasting.

<span style="color:#0000dd;">So "element-wise" just means "do this math for every element"?</span>

Yes, that's correct. "Element-wise" means performing the mathematical operation between corresponding elements of two arrays, or between the elements of a single array.

The term is used to emphasize that the operation is being performed on each individual element of the array(s), rather than on the array as a whole.

Matrix multiplication and matrix addition are ***not*** performed element-wise.

## Matrix Multiplication

For example, let's consider two matrices, A and B:

Matrix A:

    1 2
    3 4
    
Matrix B:

    5 6
    7 8

<br>
Now, let's say we want to multiply these two matrices, A and B. 

The rule is, the number of columns in the first matrix (Matrix A) should match the number of rows in the second matrix (Matrix B). Here, both matrices are 2x2 (2 rows and 2 columns), so we're good.

To multiply, we take each row of the first matrix and multiply it, element by element, with each column of the second matrix, and add up those products.

So, the first element of the result would be (1x5 + 2x7), because we're taking the first row of the first matrix and the first column of the second. 

Let's write down the calculations for each element of the resulting matrix:

    (1*5 + 2*7)  (1*6 + 2*8)
    (3*5 + 4*7)  (3*6 + 4*8)
    
<br>
Do the math:

    (5 + 14)  (6 + 16)
    (15 + 28)  (18 + 32)
    
<br>
And you get:

    19  22
    43  50
    
<br>
So, the result of multiplying matrices A and B is:

    19 22
    43 50

<br>
Just remember, to multiply matrices, we take rows from the first one and columns from the second one, multiply each element together, and add them up. And always make sure the number of columns in the first matrix matches the number of rows in the second one!

## Calculate it

You can perform matrix multiplication in using the `numpy` library, which is a popular library for handling large, multi-dimensional arrays and matrices.

Import numpy, and use it to multiply your matrices:

```python
import numpy as np

matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

result = np.dot(matrix_a, matrix_b)

print(result)
```

Here, `np.array` is used to create a numpy array (a kind of matrix) from your lists. `np.dot` is the function that performs the matrix multiplication.

When you run this code, you should get:

```
[[19 22]
 [43 50]]
```

This is the same result as we got when doing the multiplication by hand!

Starting with Python 3.5, you can also use the `@` operator for matrix multiplication of numpy arrays:

```python
result = matrix_a @ matrix_b
```

This will give you the same result. The `@` operator is more readable and intuitive if you're doing a lot of matrix operations.

<br>

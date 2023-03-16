## Normalize Pixel Values

<!-- normalization -->
What does the dot mean at the end?

```py
train_set_x = train_set_x_flatten/255.
```

It's to indicate that the division operation should be performed element-wise on the array.

In this case, `train_set_x_flatten` is likely a numpy array containing pixel values of grayscale images.

By dividing each element of the array by 255, the pixel values are normalized to be between 0 and 1, which can be beneficial for training certain types of machine learning models.

Without the dot at the end, the division operation would not be performed element-wise, and an error would occur since dividing an array by a scalar requires broadcasting.

### Remember?

So "element-wise" just means "do this math for every element"?
Why don't they just say that?

Yes, that's correct. "Element-wise" means performing the mathematical operation between corresponding elements of two arrays, or between the elements of a single array.

The term is used to emphasize that the operation is being performed on each individual element of the array(s), rather than on the array as a whole.

Matrix multiplication and matrix addition are not performed element-wise.

### Whaddya mean Beethoven wasn't so great?!

What do you mean Matrix multiplication isn't performed element-wise?

Matrix multiplication is not performed element-wise because it involves a more complex calculation than simply performing a mathematical operation between corresponding elements of two matrices.

<!--In matrix multiplication, the elements of one matrix are multiplied with the elements of another matrix according to a specific set of rules. Specifically, the product of two matrices A and B is obtained by taking the dot product of each row in matrix A with each column in matrix B.

The resulting matrix has a number of rows equal to the number of rows in matrix A and a number of columns equal to the number of columns in matrix B.

In contrast, element-wise multiplication involves multiplying corresponding elements of two matrices, resulting in a matrix of the same shape as the original matrices. Similarly, element-wise addition involves adding corresponding elements of two matrices, resulting in a matrix of the same shape as the original matrices.

So, while matrix multiplication and element-wise multiplication may both involve multiplication of matrices, they are fundamentally different operations that are performed in different ways.-->

Oh.

<br>

## List comprehension

You're creating a new list.  Like JavaScript map and/or filter.  That's all.

List = [**expression** for **item** in **iterable** **(if conditional)**]

<img src="../../images/list-comp.jpg" width="700">

```python
# List comp
a = [4, 6, 7, 3, 2]
b = [x for x in a if x > 5]
print(b)  # [6, 7]

# Equivalent for-loop:
b = []
for x in a:
    if x > 5:
        b.append(x)
print(b)  # [6, 7]
```

How to access the index in 'for' loops? Easy &ndash; enumerate.

```python
xs = [8, 23, 45]
for idx, x in enumerate(xs):
    print(idx, x)
```

<br>

## Dictionary comprehension

```py
classes = ['pizza', 'steak', 'sushi']
class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
# {'pizza': 0, 'steak': 1, 'sushi': 2}
```

A little more complex; but it's the same thing.  You read it like this:

```python
class_name: i  # This part goes together, as one thing.

# An then it's just
for i, class_name in enumerate(classes)
```

This is a dictionary comprehension in Python. It creates a dictionary where the keys are the `class_name` strings and the values are their corresponding indices in the `classes` list.

Specifically, the comprehension iterates over the `classes` list and uses the built-in `enumerate` function to get a tuple of `(index, class_name)` for each item in the list. Then, it creates a dictionary with the `class_name` as the key and the `index` as the value using the syntax `{key: value for (key, value) in iterable}`.

<br>

## JS Array Comprehension

Consider the following Python list comprehension that squares all even numbers in a list:

```py
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = [n ** 2 for n in numbers if n % 2 == 0]
print(squares)  # Output: [4, 16, 36, 64, 100]
```

This can be written in JavaScript using `Array.map()`:

Create a new array that contains the squares of all even numbers in an existing array:

```js
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let squares = numbers.filter(n => n % 2 === 0).map(n => n ** 2);
console.log(squares); // Output: [4, 16, 36, 64, 100]
```

In the example above, the `filter()` method is used to create a new array that contains only the even numbers from the `numbers` array. Then, the `map()` method is used to create a new array that contains the squares of each number in the filtered array.

<br>

## What does this mean?  `X[:, 0]`

In Python, `X[:, 0]` is a NumPy array indexing expression that selects all the rows in the array X and only the first column. This expression returns a 1-dimensional NumPy array that contains all the elements in the first column of the array X.

To break it down further:

* `X` is a NumPy array.
* `:` in the first index position means that all rows are selected.
* `0` in the second index position means that only the first column is selected.
* The resulting expression selects all rows in the first column of the array `X`.

For example, if `X` is a 2-dimensional NumPy array with shape `(3, 2)`:

```py
import numpy as np

X = np.array([
    [1, 2], 
    [3, 4], 
    [5, 6]
])
```

Then, `X[:, 0]` would be equal to:

```py
array([1, 3, 5])
```

This is a 1-dimensional array that contains all the elements in the first column of X.

<table border="1">
<tr><td width="25px;" align="center">1</td><td width="25px;" align="center">2</td></tr>
<tr><td width="25px;" align="center">3</td><td width="25px;" align="center">4</td></tr>
<tr><td width="25px;" align="center">5</td><td width="25px;" align="center">6</td></tr>
</table>

<br>

## More Indexing 🧱 🏃🏻‍♀️

```py
X[:pos] # means: Get all elements up until `position`.
X[pos:] # means: Get all elements from the position, onwards.
```

<br>
Think of it like running until you hit a wall.  The wall is the colon and the position.

`[pos:]` means the wall is behind you.

`[:pos]` means run *into* the wall.

<br>

## Say, "Shape" 😁

A (3, 2) array would typically be referred to as a "three by two array".

<span style="color:brown;">It's usually width x height.</span>

<span style="color:brown;">But this right here is height x width.</span> 🤠

<span style="color:brown;">A cowgirl's height is more important than her girth (spits tobacco)...</span>

<span style="color:brown;">So that's rows (how tall), and columns (how wide).</span> <span style="font-size:27px;">🐎</span>

This is because the first number (3) represents the number of rows in the array, and the second number (2) represents the number of columns.

So, a (3, 2) array would have 3 rows and 2 columns.

You can also think of it as a **rectangular grid** with 3 rows and 2 columns, where each element in the grid corresponds to a value in the array.

<table border="1">
<tr><td width="25px;">&nbsp;</td><td width="25px;">&nbsp;</td></tr>
<tr><td width="25px;">&nbsp;</td><td width="25px;">&nbsp;</td></tr>
<tr><td width="25px;">&nbsp;</td><td width="25px;">&nbsp;</td></tr>
</table>

<br>

## shape = ()

An array with a shape of `()` is called a **scalar** or a 0-dimensional array.

It represents **a single value**, such as a number or a boolean.

<span style="color:red;">1</span>

On the other hand, an array with a shape of `(1)` is a one-dimensional array with a single element.

<span style="color:red;">[1]</span>

So, `()` and `(1)` are not the same. A scalar array is different from a one-dimensional array with a single element.

```python
import numpy as np

# Example 1
scalar_arr = np.array(5)
print(scalar_arr)  # 5 - notice 5 is the value, not the array length
print(scalar_arr.shape)  # ()

# Example 2
one_dim_arr = np.array([5])
print(one_dim_arr)  # [5]
print(one_dim_arr.shape)  # (1,)
```

<br>

## NumPy Array Reshaping

### array.reshape()

What does it mean, using numpy, when someone does

```py
array.reshape(-1, 1)
```

<span style="color:#0000dd;font-size:larger;">-1 tells the tensor to calculate the size of the dimension in that place.</span>

In NumPy, the `reshape()` function is used to change the shape (dimensionality) of an array. The argument **-1** in `array.reshape(-1, 1)` is a **placeholder** for an unknown dimension size, which is inferred by NumPy based on the total number of elements in the array.

When -1 is passed as the first argument to `reshape()`, NumPy **automatically calculates the number of rows** required for the reshaped array based on the size of the second argument, which in this case is 1. Therefore, `array.reshape(-1, 1)` reshapes the input array to a column vector (i.e., a 2D array with a single column) while keeping the original number of rows.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # shape (3, 3)
b = a.reshape(-1, 1)  # shape it to "idk" rows and 1 column

print(b.shape)  # (9, 1)
```

## See also:

* "TensorFlow/TensorFlow.js/Model Input Shape Mismatch.md"
* PyTorch/notes/dummy.md
* docs/Display.md

<br>

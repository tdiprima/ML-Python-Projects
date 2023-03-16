## What does this mean?  `X[:, 0]`

In Python, `X[:, 0]` is a NumPy array indexing expression that selects all the rows in the array X and only the first column. This expression returns a 1-dimensional NumPy array that contains all the elements in the first column of the array X.

To break it down further:

* `X` is a NumPy array.
* `:` in the first index position means that all rows are selected.
* `0` in the second index position means that only the first column is selected.
* The resulting expression selects all rows in the first column of the array `X`.

For example, if `X` is a 2-dimensional NumPy array with shape `(3, 2)`:

```py
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

### X[:, 0] = everything, in the 0th column

*Everything in every row, that is.*

You can say `X[:, 0]` as "all the rows of X, but only the values in the first column".

The `:` before the comma (`:`) means that we select all rows.

And 0 after the comma means that we select only the first column.

So, the resulting expression selects all rows of the X array, but only returns the values in the first column.

## Say, "Shape".

How do I say, in words, a (3, 2) array?

In words, a (3, 2) array would typically be referred to as a "three by two array".

<span style="color:brown;">It's usually width x height.</span>

<span style="color:brown;">But this right here is height x width. 🤠</span>

<span style="color:brown;">A cowboy's height is more important than his girth (spits tobacco)...</span>

<span style="color:brown;">So that's rows (how tall), and columns (how wide).</span> <span style="font-size:27px;">🐎</span>

This is because the first number (3) represents the number of rows in the array, and the second number (2) represents the number of columns.

So, a (3, 2) array would have 3 rows and 2 columns.

You can also think of it as a **rectangular grid** with 3 rows and 2 columns, where each element in the grid corresponds to a value in the array.

<!--<table border="1">
<tr><td width="25px;">&nbsp;</td><td width="25px;">&nbsp;</td></tr>
<tr><td width="25px;">&nbsp;</td><td width="25px;">&nbsp;</td></tr>
<tr><td width="25px;">&nbsp;</td><td width="25px;">&nbsp;</td></tr>
</table>-->

## More Indexing

```py
X[:pos] # means: Get all elements up until `position`.
X[pos:] # means: Get all elements from the position, onwards.
```

<br>
Think of it like running until you hit a wall.  The wall is the colon and the position.

`[pos:]` means the wall is behind you.

`[:pos]` means run *into* the wall.

<br>

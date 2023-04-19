## Linear regression plot

```python
import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]

# Plot the points
plt.plot(x, y, 'ro')

# Set the axis ranges
plt.axis([0, 6, 0, 20])

# Draw line of best fit
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()
```

## What's this?

```c
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
```

Let's say you have two lists of numbers, `x` and `y`, and you want to see if there is a straight line that goes through all the points on a graph.

The code `np.polyfit(x, y, 1)` calculates the equation of a **straight line** that best fits the data in `x` and `y`.

Then, the code `np.poly1d(np.polyfit(x, y, 1))` turns that equation into a **function** that you can use to calculate the `y` value for any `x` value along the line.

Next, `np.unique(x)` makes a **new list** with just the **unique values** of `x` in it (in other words, if there are any repeated values of `x`, they will only appear once in this new list).

Finally, `plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))` uses the function we created earlier to **calculate the `y` values** for each **unique `x`** value, and then **plots** those points on a graph as a **line**.

So, when you run this code, you'll see a line on a graph that goes through all the points in `x` and `y` as closely as possible.

<!--
This code does the following:

1. `np.unique(x)` returns an array of unique values from the list `x`.

1. `np.polyfit(x, y, 1)` fits a first-degree polynomial (linear regression) to the data `points (x[i], y[i])` in the lists `x` and `y`, returning the coefficients of the polynomial in reverse order.

1. `np.poly1d(np.polyfit(x, y, 1))` constructs a polynomial object based on the coefficients from step 2, representing the linear regression line.

1. `(np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))` applies the polynomial to the unique values in `x` obtained in step 1, returning the corresponding `y` values that fall on the linear regression line.

1. Finally, `plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))` plots the `x` values from step 1 on the x-axis and the `y` values from step 4 on the y-axis, resulting in a line plot of the linear regression line that best fits the data in x and y.

In summary, this code performs a linear regression on the data points in `x` and `y` and plots the resulting line using Matplotlib's plot function.

#rufkm
-->

<br>

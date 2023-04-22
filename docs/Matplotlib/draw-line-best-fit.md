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

```py
plt.plot(unique(x), poly1d(polyfit(x, y, 1))(unique(x)))
```

Let's say you have two lists of numbers, `x` and `y`, and you want to see if there is a straight line that goes through all the points on a graph.

The code `polyfit(x, y, 1)` calculates the equation of a **straight line** that best fits the data in `x` and `y`.

Then, the code `poly1d(polyfit(x, y, 1))` turns that equation into a **function** that you can use to calculate the `y` value for any `x` value along the line.

Next, `unique(x)` makes a **new list** with just the **unique values** of `x` in it (in other words, if there are any repeated values of `x`, they will only appear once in this new list).

Finally, `plt.plot(unique(x), poly1d(polyfit(x, y, 1))(unique(x)))` uses the function we created earlier to **calculate the `y` values** for each **unique `x`** value, and then **plots** those points on a graph as a **line**.

So, when you run this code, you'll see a line on a graph that goes through all the points in `x` and `y` as closely as possible.

## Numpy Linear Equation

<span style="color:#0000dd;">Numpy's polyfit() calculates the equation of a straight line that best fits the data in x and y.  The result that I get is a numpy array: [ 4.7, -4.55 ].  <span style="color:#0000dd;font-weight:bold;">How is this an equation?</span></span>

The `polyfit` function in NumPy can fit a polynomial function of degree `n` to a set of data points `(x, y)`. When `n=1`, it fits a linear function to the data. The result of the `polyfit` function in your case, a numpy array `[4.7, -4.55]`, **represents the coefficients of the linear function** that best fits your data. 

In particular, the coefficients correspond to the **slope** and **y-intercept** of the line, respectively. So the equation of the line can be written as:

```py
y = slope * x + y_intercept
```

Substituting the values you obtained from `polyfit`, the equation becomes:

```py
y = 4.7 * x - 4.55
```

Therefore, the numpy array `[4.7, -4.55]` represents the equation of the line that best fits your data.

# Dots and line

<span style="color:#0000dd;">So when we plot using matplotlib, do those values represent the dots in the graph, and then matplotlib automatically draws the line though them?</span>

Yes, that's correct!

```py
plt.plot(unique(x), poly1d(polyfit(unique(x)))
```

creates a line plot with the x-axis values as `unique(x)` and the y-axis values as `poly1d(polyfit(x, y, 1))(unique(x))`. 

The function `poly1d(polyfit(x, y, 1))` returns a polynomial function of degree 1 (i.e., a linear function) that best fits the data `(x, y)`.

The `polyfit` function calculates the coefficients of the linear function.

The `poly1d` function creates a callable object from these coefficients that can be used to evaluate the function at any x-value.

By passing `unique(x)` as the argument to this callable object, you evaluate the fitted function at each of the unique values in `x`. The resulting y-values represent the points on the line that will be plotted. 

When you plot these points using `plt.plot(unique(x), poly1d(polyfit(x, y, 1))(unique(x)))`, matplotlib automatically connects the dots with a straight line, giving you the best-fit line that passes through the points.

Gweat.

## What's a polynomial function?

A **polynomial function** is a type of mathematical function that is used to describe many real-world phenomena, like the growth of populations, the **trajectory** of objects, and the shapes of curves.

In its simplest form, a polynomial function is just an expression made up of numbers, variables (usually x), and exponents. For example, the function:

```py
f(x) = 2x^2 - 3x + 1

# 2x squared
```

is a polynomial function of degree 2, which means it has a highest exponent of 2. 

When we evaluate this function at different values of x, **we get a set of y-values that form a curve.** The shape of this curve depends on the coefficients in the polynomial function. 

Polynomial functions are really useful because they can be used to approximate more complex functions or data sets. We can fit a polynomial function to a set of data points using a process called **polynomial regression**, which finds the best-fitting polynomial curve that passes through the data points. 

In the case of

```py
polyfit(x, y, 1)
```

we're using a polynomial function of **degree 1**, which means it's a **linear** function. This is a simple way to approximate a straight line that best fits our data points.

<br>

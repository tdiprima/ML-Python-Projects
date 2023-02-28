## numpy.log()

`log()` is a function in the NumPy library that computes the **natural logarithm** of a given input array or value.

The natural logarithm, denoted as `ln(x)`, is the logarithm to the base `e`, where `e` is a mathematical constant approximately equal to `2.71828`.

```py
import numpy as np

# Compute the natural logarithm of a single value
x = 10
result = np.log(x)
print(result)  # 2.302585092994046

# Compute the natural logarithm of each element in an array
arr = np.array([1, 2, 3, 4, 5])
result = np.log(arr)
print(result)  # [0.         0.69314718 1.09861229 1.38629436 1.60943791]
```

<br>
<!-- I'd be happy to explain natural logarithms and base e in a way that's easy to understand! -->

**Logarithms** are a way to measure how many times you need to multiply a number by itself to get another number.

For example, if you multiply 2 by itself three times (2 x 2 x 2), you get 8.

So the logarithm of 8 with base 2 is **3**, because you needed to multiply 2 by itself three times to get 8.

The **natural logarithm** is a special kind of logarithm that uses a special number called "e" as its base.

The value of `e` is about `2.71828`.

So if you wanted to find the natural logarithm of a number, you would figure out how many times you need to multiply `e` by itself to get that number.

Natural logarithms are especially useful in situations where you need to **measure things that grow or change exponentially,** like the spread of a virus or the growth of a population. <span style="font-size:27px;">👻</span>


### Are logarithm and getting the n-root of a number related to each other?

Yes, logarithms and taking roots are related to each other, and in fact, they are inverse operations.

To see why, consider the following:

If we take the logarithm of a number x to a base b, we get the power to which we need to raise b to obtain x:

log_b(x) = p

In other words, we are asking: "What power p must we raise b to, in order to obtain x?".

If we then take b to the power of p, we get:

b^p = x

This is the same as taking the p-th root of x:

x^(1/p) = b

So we can see that taking the logarithm of a number to a certain base, and taking the p-th root of that number, are inverse operations. One "undoes" the effect of the other.

For example, if we take the logarithm of 100 to base 10, we get:

log_10(100) = 2

This means that we need to raise 10 to the power of 2 in order to obtain 100. If we then take the square root of 100, we get:

sqrt(100) = 10^(1/2) = 10^0.5 = 3.1623...

This is the same as raising 10 to the power of 2 divided by 2, which "undoes" the effect of taking the logarithm to base 10.

So in summary, logarithms and taking roots are related to each other, and we can use one operation to "undo" the effect of the other.

### Code example using logarithms and roots

```py
import numpy as np

# Compute the natural logarithm of a value
x = 10
log_x = np.log(x)
print("The natural logarithm of", x, "is", log_x)  # 2.302585092994046

# Compute the square root of a value
y = 25
sqrt_y = np.sqrt(y)
print("The square root of", y, "is", sqrt_y)  # 5.0

# Compute the logarithm of a value with a base of 2
z = 8
log_z_base2 = np.log2(z)
print("The logarithm of", z, "with base 2 is", log_z_base2)  # 3.0
```

<br>

**numpy.log()** = compute the natural logarithm

**np.sqrt()** = compute the square root

**np.log2()** = compute the logarithm of a number, with a base of 2

i.e. How many times we need to multiply 2 by itself to get 8.

**`(2 x 2 x 2) = 8`**

### Why would we ever want the natural logarithm?  It seems kinda dumb &ndash; 2.71828!

That's a good question! While the number "e" (approximately 2.71828) that is used as the base for natural logarithms may seem like a strange choice at first, it turns out that the natural logarithm has some very useful properties that make it a valuable tool in many areas of mathematics and science.

One reason why the natural logarithm is important is that it has a close relationship with exponential functions. Specifically, if we take the natural logarithm of an exponential function, we get a straight line. This makes it easier to analyze and understand exponential growth or decay in various contexts.

The natural logarithm also shows up frequently in probability theory, where it plays a central role in the calculation of probabilities and in the formulation of many important theorems.

Furthermore, the natural logarithm is often used in calculus to simplify complicated expressions and make them easier to work with. For example, it can be used to express the derivative of certain functions in a more concise and elegant way.

Overall, while the number "e" and the concept of natural logarithms may seem a bit abstract or esoteric at first, they are actually extremely useful tools in many areas of mathematics, science, and engineering.

### Using the natural logarithm and Matplotlib to plot a graph of an exponential function and its natural logarithm

<!--You mentioned that if we take the natural logarithm of an exponential function, we get a straight line.  Can you give me a python example, using natural logarithm, to create a graph, using matplotlib?-->

```py
import numpy as np
import matplotlib.pyplot as plt

# Define the exponential function
def exp_func(x):
    return np.exp(x)

# Define the domain of the function
x_values = np.linspace(-5, 2, 100)

# Compute the values of the function and its logarithm
y_values = exp_func(x_values)
log_y_values = np.log(y_values)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the exponential function on the first subplot
ax1.plot(x_values, y_values, label='exp(x)')
ax1.legend()

# Plot the natural logarithm of the function on the second subplot
ax2.plot(y_values, log_y_values, label='log(exp(x))')
ax2.legend()

# Add titles and axis labels to the subplots
ax1.set_title('Exponential Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2.set_title('Natural Logarithm of Exponential Function')
ax2.set_xlabel('exp(x)')
ax2.set_ylabel('log(exp(x))')

# Show the plot
plt.show()

```

<br>

In this code, we first define a simple exponential function using `np.exp()`.

We then define a range of x values using `np.linspace()`, and compute the corresponding y values for the function and its natural logarithm using `exp_func()` and `np.log()`.

Next, we create a figure with two subplots using `plt.subplots()`.

We plot the exponential function on the first subplot using `ax1.plot()`, and plot the natural logarithm of the function on the second subplot using `ax2.plot()`.

Finally, we add titles and axis labels to the subplots using `ax.set_title()`, `ax.set_xlabel()`, and `ax.set_ylabel()`, and display the plot using `plt.show()`.

When you run this code, you should see a plot with two subplots side by side, one showing the **exponential function** and the other showing its **natural logarithm.**

You'll notice that the **second plot** has a **linear relationship** between the x and y values, as we mentioned earlier.

<br>

## Geometric mean

The **geometric mean** is another type of average that's commonly used in mathematics and statistics.

It is calculated by taking the **nth root** of the **product of n numbers**.

It is often used to find the **average growth rate** or **compound interest rate** over a set of numbers.

Let's say we want to find the geometric mean of the numbers 2, 4, 6, and 8:

```py
# Import the math module
import math

# Define a list of numbers
numbers = [2, 4, 6, 8]

# Calculate the product of the numbers
product = 1
for num in numbers:
    product *= num

# Calculate the geometric mean
geometric_mean = math.pow(product, 1/len(numbers))

# Print the result
print("The geometric mean is:", geometric_mean)
```

<br>
In this example, the math module is imported to use the `pow()` function, which takes the **nth root** of a number.

The product of the numbers is calculated using a for loop that multiplies each number together.

Then, the geometric mean is calculated by taking the product to the power of `1/n`, where `n` is the number of numbers in the list.

### why is it called geometric when there are no shapes involved? 🟢 🟦 ♦️

The term "geometric" in "geometric mean" comes from the fact that it is related to the **calculation** of **lengths, areas, and volumes** in geometry.

If you have two positive numbers `a` and `b`, the arithmetic mean of a and b is `(a + b) / 2`.

The geometric mean of a and b is the **square root** of `ab`.

The square root of ab represents the length of the side of a square whose area is equal to the product of a and b.

In higher dimensions, the geometric mean can be thought of as the **nth root** of the **product of n numbers**, and it can be related to the **volume** of a hypercube whose side lengths are given by the n numbers.

So while the geometric mean may not involve actual shapes like squares or cubes, it is related to the **calculation of geometric quantities** and has a geometric interpretation in terms of lengths, areas, and volumes.

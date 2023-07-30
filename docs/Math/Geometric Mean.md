## Geometric mean <span style="font-size: 27px;">📐 ⨏

The geometric mean is another type of average that's commonly used in mathematics and statistics.

It is calculated by taking the nth root of the product of n numbers.

It is often used to find the average growth rate or compound interest rate over a set of numbers.

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

In this example, the `math` module is imported to use the **`pow()`** function, which takes the **nth root** of a number.

The product of the numbers is calculated using a for loop that multiplies each number together.

Then, the geometric mean is calculated by taking the product to the power of `1/n`, where `n` is the number of numbers in the list.

<span style="color:#880000;font-size:larger;">Geometric Mean is calculated by taking the nth root of the product of n numbers.</span>

First, let's talk about what a "mean" is. A mean is a type of average. It's a way of figuring out what the "typical" value is when you have a bunch of different numbers.

There are different types of means, but the most common one you've probably heard of is the "arithmetic mean." That's the one you get by adding up all the numbers and then dividing by how many numbers there are.

The geometric mean is a different type of mean. To calculate the geometric mean, you need to multiply all the numbers together, and then take the nth root of the result.

### Example

Let's say you have three numbers: 2, 4, and 8.

To calculate the geometric mean of these numbers, you first multiply them together: `2 x 4 x 8 = 64`.

```js
let arr = [ 2, 4, 8 ]
let product = 1;

for (let i = 0; i < arr.length; i++) {
  product *= arr[i];
}

console.log(product);
```

Or:

```js
// JavaScript Array reduce() Example
let array = [ 2, 4, 8 ];

let pro = array.reduce(function(a, b) {
  return a * b;
});

console.log(pro);
```

<br>
Next, you take the nth root of that result. 

The "n" in this case is the number of numbers you're working with, which is **3** in this example.

So you take the cube root of 64, which is 4.

```js
let len = arr.length;

// Math.pow(n, 1/root);
// Math.pow(64, 1/3);
let result = Math.pow(product, 1/len);

// We already know we want the cube root:
Math.cbrt(64);
```

<br>
So, the geometric mean of the numbers 2, 4, and 8 is 4.

That means that **4** is the "typical" value when you're looking at these numbers using the geometric mean.

## Why is it called geometric?

The term "geometric" in "geometric mean" comes from the fact that it is related to the <span style="color: green; font-size: larger">calculation of lengths, areas, and volumes in geometry.</span>

If you have two positive numbers `a` and `b`, the arithmetic mean of `a` and `b` is

<span style="color: blue; font-size: larger">(a + b) / 2</span>

The geometric mean of `a` and `b` is

<span style="color: blue; font-size: larger">the square root of ab</span>

The square root of `ab` represents the length of the side of a square whose area is equal to the product of `a` and `b`.

In higher dimensions, the geometric mean can be thought of as the nth root of the product of `n` numbers.

And it can be related to the volume of a hypercube whose side lengths are given by the `n` numbers.

So while the geometric mean may not involve actual shapes like squares or cubes, it is related to the calculation of geometric quantities and has a geometric interpretation in terms of lengths, areas, and volumes.

## Funky Tools 🌈 🛼

<span style="color:#0000dd;">Is there a function for calculating the product of a list of numbers?</span>

There isn't a built-in Python function that directly calculates the product of a list of numbers. However, you can use Python's `functools.reduce()` function in combination with the `operator.mul` function to achieve this. Here's how:

```python
import functools
import operator

numbers = [2, 4, 6, 8]
product = functools.reduce(operator.mul, numbers, 1)

print(product)
```

<br>

In this code, `functools.reduce()` applies `operator.mul` (which is the multiplication operation) pairwise to all elements in the `numbers` list in a cumulative way.

The third argument `1` is the initializer, which is optional. It's used as the initial value in the computation. If the list `numbers` is empty, `1` will be returned.

Alternatively, you can use a simple `for` loop to calculate the product (like before).

Both of these will output `384`, which is the product of 2, 4, 6, and 8.

<br>

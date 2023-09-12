## Epsilon: Small numbers matter

Epsilon (**&#949;**) represents a very small number that's close to zero, but not quite zero.

In math, we use epsilon to talk about how close two numbers are to each other. If two numbers are really close, but not exactly the same, we can say they're "epsilon apart".

<br>
For example, let's say you have two numbers: 3 and 3.0001. These numbers are really close, but not exactly the same. We could say they're epsilon apart, because the difference between them is very small (less than epsilon).

In science, epsilon is often used to talk about errors or uncertainties in measurements. When we measure something, it's not always possible to get the exact value. There can be small errors or uncertainties that creep in. We use epsilon to talk about how much wiggle room there is in our measurements.

## Python

We can use epsilon as a way to compare two numbers and check if they're "close enough" to each other.

We can do this using the **`abs`** function, which gives us the **absolute value** (i.e., the distance from zero) of a number.

```python
a = 3.0
b = 3.0001
threshold = 0.001  # Has to be a bigger, small number.

diff = abs(a - b)
result = diff <= threshold

print(result)  # True
```

<br>
Epsilon as 0.0001 (1e-4) seems to be very popular.

In some cases, you might want to use a larger or smaller value of `epsilon` depending on the precision you need in your calculations.

There may be other ways to use epsilon to check for closeness or to handle errors and uncertainties in measurements:

```py
import math

def are_epsilon_apart(a, b, epsilon):
    return math.isclose(a, b, abs_tol=epsilon)  # is close
```

## JavaScript

```js
a = 1.23456789
b = 1.23456777

threshold = 0.000001

diff = Math.abs(a - b); // Absolute value of difference
console.log(diff < threshold); // True if difference is less than threshold
```

## 🆚 Euler's number

Oh, so you *set* epsilon.  I thought epsilon was a numeric constant, like **`e`** (approximately equal to `2.71828`).

Great question! You're right that `e` is a numeric constant (known as **Euler's number**), but **epsilon** is actually a variable that you can set to any value you want, depending on the specific problem you're trying to solve.

<br>

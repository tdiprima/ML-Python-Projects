## In 10 words or less, what is the "harmonic mean"?

A **type of average** that emphasizes **small** values.

### In very simple terms, what is the "harmonic mean"?

In very simple terms, the harmonic mean is a type of average that is used to calculate a central or representative value for a set of numbers.

It is calculated by dividing the **number of values in the set** by the sum of the reciprocal of each value in the set.

The **reciprocal** of a number is simply 1 divided by the number.

For example, the reciprocal of `4` is `1/4`, and the reciprocal of `5` is `1/5`.

## Calculate the harmonic mean

To calculate the harmonic mean of a set of numbers:

1. calculate the reciprocal of each number
2. take the sum of those reciprocals
3. divide the number of values in the set by the sum of the reciprocals.

# 📈 🚙

The harmonic mean is often used to calculate **averages** for **rates or ratios**, such as the average speed of a journey or the average price-to-earnings ratio of a stock market index.

It is generally **less affected by extreme values** than other types of averages, such as the arithmetic mean or the geometric mean.

## Python code example

Let's say we want to find the harmonic mean of the numbers 2, 4, 6, and 8:

```py
# Define a list of numbers
numbers = [2, 4, 6, 8]

# Calculate the reciprocal of each number
reciprocals = [1/x for x in numbers]

# Calculate the average of the reciprocals
average = sum(reciprocals) / len(reciprocals)

# Take the reciprocal of the average
harmonic_mean = 1 / average

# Print the result
print("The harmonic mean is:", harmonic_mean)
```

<br>
In this example, the reciprocal of each number is calculated using a **list comprehension**, which is a way to create a new list by applying an operation to each element of an existing list.

The **`sum()`** function is used to find the sum of the reciprocals, and the **`len()`** function is used to find the number of reciprocals in the list.

The **average** is then calculated by dividing the sum by the number of elements in the list.

Finally, the **reciprocal of the average** is taken to find the harmonic mean.

This is because the harmonic mean is the reciprocal of the arithmetic mean of the reciprocals.

## Why?

Harmonic mean is useful for averaging rates or ratios, like speed or fuel efficiency.

### So it's used for continuous variables? 🤔

*Like, if I see a continuous variable, this is the type of averaging method I should use?*

Not necessarily. The harmonic mean is generally used when you want to find the average of rates or ratios, regardless of whether the variables are continuous or discrete.

For example, the harmonic mean could be used to find the **average speed of a car over a journey** with varying speeds.

In this case, the **speeds are continuous variables.**

But the harmonic mean could also be used to find the **average price-to-earnings ratio** of a set of stocks, which is a **discrete variable.**

However, for **data with outliers**, the harmonic mean can be strongly influenced by them and may not provide a representative measure of the central tendency.

In such cases, other measures like the arithmetic mean or median may be more appropriate.

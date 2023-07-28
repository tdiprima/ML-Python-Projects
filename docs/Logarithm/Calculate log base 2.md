## Binary Numbers & Logs

<span style="color:#0000dd;">WTH is "base 2" and "logarithm base 2 of 8"?</span>

### Base 10

You're familiar with counting from 1 to 10, right? We do this in what we call "base 10". It's called base 10 because we have 10 digits from 0 to 9 to represent numbers. After 9, we start over at 0 and put a 1 in front of it to make 10. We keep going like this, using combinations of these 10 digits to represent all numbers.

### Base 2

Now, imagine if you only had 2 digits, 0 and 1, instead of 10. This is what we call "base 2", or binary. Just like with base 10, we use combinations of these 2 digits to represent all numbers. The first few numbers in base 2 look like this:

**Base 10:**  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10...

**Base  2:**  0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010...

So, in base 2, "10" means what we think of as 2, "11" is 3, "100" is 4, and so on.

### Logarithm base 2 of 8

A logarithm is a way to ask the question "how many of this number do I need to multiply together to get that number?"

So we're asking "how many 2's do we need to multiply together to get 8?"

Let's try it out:

- 2 (this is just one 2, so it's not enough)
- 2 x 2 = 4 (this is two 2's, but it's still not enough)
- 2 x 2 x 2 = 8 (this is three 2's, and we've got it!)

So, the logarithm base 2 of 8 is 3, because we need to multiply three 2's together to get 8. 

## Calculate in Python

You can use the `math` module in Python to calculate the logarithm of a number.

```py
import math

log_value = math.log2(8)

print(log_value)
```

This will output `3.0`, which is the logarithm base 2 of 8.


## Calculate in JavaScript

You can use the `Math.log2()` method in JavaScript to calculate the logarithm of a number to the base 2.

```js
let logValue = Math.log2(8);

console.log(logValue);
```

This will output `3`, which is the logarithm base 2 of 8. 

Note that the `Math.log2()` method returns the result as an integer instead of a floating-point number, so it may not be an exact value.

## Calculate in R

You can calculate the logarithm of a number to a specific base using the `log()` function with the appropriate base as the second [argument].

```r
log2(8)
```

## Meaning

"Log base 2 of 8" means the exponent you need to raise 2 to, to get 8 as the result.

In other words: **2 to the power of what equals 8?**

log₂(8) = x

Here, x is the exponent that we need to find.

To solve this equation, we need to ask ourselves, "What power of 2 equals 8?"

We know that 2³ (2 raised to the power of 3) is equal to 8.

Therefore, log base 2 of 8 is equal to 3.

<br>
<span style="color:#0000dd;font-size:larger;">log₂(2x4) just means multiply 2x4</span> <span style="font-size:27px;">🤦🏻‍♀️</span>

log₂(2x4) = the logarithm with base 2 of the quantity 2x4.

log₂(2x4) = log₂(8) = 3

See?

```py
import math

x = 2 * 4  # Is just 8.
result = math.log(x, 2)
print(result)  # 3.0
```

<br>

<span style="color:#0000dd;font-size:larger;">There is no value for log2(0)</span> <span style="font-size:27px;">🙅🏻‍♀️</span>

This is because there is no exponent, x, for 2 such that... 2<sup>x</sup> = 0

```py
# Python
math.log(0, 2)  # Error
```

```r
# R
log2(0) # -Inf or -Infinity
```

<br>
[Square roots explained Bob Ross style](https://youtu.be/ocSq6v0Mmt0)

[Logs (logarithms), Clearly Explained](https://youtu.be/VSi0Z04fWj0)

<br>

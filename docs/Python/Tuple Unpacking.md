## Python: Tuple Unpacking

A tuple is a collection of ordered, immutable elements, similar to a list.

When you define a tuple, you use parentheses instead of square brackets:

```py
my_tuple = (1, 2, 3)
```

One cool thing you can do with tuples in Python is called "Tuple Unpacking." This means you can assign the values of a tuple to separate variables all at once.

### Example 1

Let's say you have a tuple of two values, and you want to assign those values to two separate variables:

```py
my_tuple = (10, 20)
a = my_tuple[0]
b = my_tuple[1]
```

With tuple unpacking, you can do the same thing in a single line of code:

```py
a, b = my_tuple
```

The first variable (`a`) will be assigned the value of the first element in `my_tuple` (which is `10`).

The second variable (`b`) will be assigned the value of the second element in `my_tuple` (which is `20`).

### Example 2

You can even use tuple unpacking to swap the values of two variables without needing a third variable.

```py
a = 1
b = 2
a, b = b, a  # No parens; it's not a list.
print(a) # 2
print(b) # 1
```

In this example, we're swapping the values of `a` and `b`.

The right-hand side of the = operator creates a tuple `(b, a)` with the values of `b` and `a` in reverse order.

The left-hand side of the `=` operator uses tuple unpacking to assign the values of the tuple to `a` and `b` in reverse order, effectively swapping their values.

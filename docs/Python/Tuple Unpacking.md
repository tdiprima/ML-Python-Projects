## Python: Tuple Unpacking

A tuple is a collection of ordered, immutable elements, similar to a list.

When you define a tuple, you use parentheses instead of square brackets:

```py
my_tuple = (1, 2, 3)
```

<br>
"Tuple Unpacking" means you can assign the values of a tuple to separate variables all at once.

### Tuple Unpacking

```py
my_tuple = (10, 20)
a, b = my_tuple
```

### Variable swap

You can even use tuple unpacking to swap the values of two variables without needing a third variable.

```py
a = 1
b = 2
a, b = b, a
```

<br>

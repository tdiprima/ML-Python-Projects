## Function annotation

```py
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)
```

The notation with the arrow `->` in the function signature is called a function annotation.

It is used to **specify the type** of the function's input arguments and return value.

## Type hinting

How do I indicate what data types the function parameters should be?

You can indicate the expected data types for function parameters using **type hints** in Python.

Type hints are annotations added to function arguments and return values that indicate the expected data types of these values.

To specify a type hint for a parameter in a function definition, you can add a colon after the parameter name followed by the expected data type.

```python
def add_numbers(x: int, y: int) -> int:
    return x + y
```

In this example, the `add_numbers` function takes two integer arguments (`x` and `y`) and returns an integer value.

The `int` after the parameter names and before the `->` indicates that the expected data type for these parameters is an integer.

You can use various data types in type hints such as `str`, `float`, `bool`, `list`, `tuple`, `dict`, and custom classes.

### FYI

Type hints are **not enforced** by the Python interpreter, but they can be helpful for documenting code and catching potential type errors during development using tools like static type checkers or IDEs that support them.

<br>

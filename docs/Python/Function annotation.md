## Function annotation

```py
import torch

def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)
```

The notation with the arrow `->` in the function signature is called a function annotation.

It is used to specify the type of the function's input arguments and return value.

## Type hinting

<span style="color:#0000dd;">How do I indicate what data types the *parameters* should be?</span>

You can indicate the expected data types for function parameters using type hints.

Type hints are annotations added to function arguments and return values that indicate the expected data types of these values.

To specify a type hint for a parameter in a function definition, you can add a colon after the parameter name followed by the expected data type.

```python
def add_numbers(x: int, y: int) -> int:
    return x + y
```

<br>

In this example, the `add_numbers` function takes two integer arguments (`x` and `y`) and returns an integer value.

The `int` after the parameter names and before the `->` indicates that the expected data type for these parameters is an integer.

You can use various data types in type hints such as `str`, `float`, `bool`, `list`, `tuple`, `dict`, and custom classes.

### FYI

Type hints are **not enforced** by the Python interpreter, but they can be helpful for documenting code and catching potential type errors during development using tools like static type checkers or IDEs that support them.

## Default values

```python
import matplotlib.pyplot as plt

# Plots training data, test data and compares predictions.
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


plot_predictions()
```

<br>
Here's one way you could refactor your function to make it more extensible and separate it into its own file.

First, let's create a new file named `plotting.py`. In this file, we'll define a new function called `plot_predictions`:

```python
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    # Do stuff...
```

<br>

**Note that we've removed the default values** for the function arguments. This is because we don't want to tie the implementation of the function to specific data sets.

Now, let's modify the original file that was using the `plot_predictions` function. We'll import the new `plot_predictions` function from `plotting.py`:

```python
from plotting import plot_predictions

# Call the function with your data sets
plot_predictions(X_train, y_train, X_test, y_test)
```

<br>
By importing the function from a separate file, we can reuse it across multiple files and projects without duplicating code.

## Oops #1

Note!  `predictions` is still a keyword.  So if you wanna pass predictions to the function, you're gonna have to use that specific keyword.

```python
plot_predictions(predictions=y_preds)
```

## Oops #2

plot_predictions() missing 4 required positional arguments: 'train_data', 'train_labels', 'test_data', and 'test_labels'

Fine. <span style="font-size:27px;">🐶</span>

So now we know that it's actually kinda suave to use the default values, because then they're ***all optional!***

<br>

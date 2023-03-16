<!--How do I import a local python file into another python file?

You can import a local Python file into another Python file by using the import statement. Here's an example:-->

Suppose you have a file called `my_module.py` in the same directory as your main file, and it contains the following code:

```python
def my_function():
    print("Hello from my_function!")
```

To import this module into your main file, you can use the following code:

```python
import my_module

# call the function from the imported module
my_module.my_function()
```

In this example, `import my_module` imports the `my_module.py` file into your main file, and `my_module.my_function()` calls the `my_function()` function from the imported module.

If you only need to import specific functions or classes from the module, you can use the following syntax:

```python
from my_module import my_function, MyClass

# call the function from the imported module
my_function()

# create an instance of MyClass from the imported module
obj = MyClass()
```

In this example, `from my_module import my_function, MyClass` imports only the `my_function()` function and the `MyClass` class from the `my_module.py` file. You can then call the imported functions and classes as usual.

### The problem is, what does this function signature mean?

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

Certainly, I'd be happy to help! Here's one way you could refactor your function to make it more extensible and separate it into its own file.

First, let's create a new Python file named `plotting.py`. In this file, we'll define a new function called `plot_predictions`:

```python
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    # Do stuff...
```

<br>

<mark>**Note that we've removed the default values**</mark> for the function arguments. This is because we don't want to tie the implementation of the function to specific data sets.

Now, let's modify the original file that was using the `plot_predictions` function. We'll import the new `plot_predictions` function from `plotting.py`:

```python
from plotting import plot_predictions

# Call the function with your data sets
plot_predictions(X_train, y_train, X_test, y_test)
```

By importing the function from a separate file, we can reuse it across multiple files and projects without duplicating code.

I hope this helps! Let me know if you have any further questions.

<span style="color:#0000dd;">You're a genius!  Thanks! :)</span>

You're welcome! I'm glad I could help. Don't hesitate to ask if you have any more questions in the future!

## Oops #1

Note!  `predictions` is a keyword.  So if you wanna pass predictions to the function, you're gonna have to use that specific keyword.

```python
plot_predictions(predictions=y_preds)
```

## Oops #2

`plot_predictions() missing 4 required positional arguments: 'train_data', 'train_labels', 'test_data', and 'test_labels'`

Fine. <span style="font-size:27px;">🐶</span>

So now we know that it's actually kinda Rico Suave to use the default values, because then they're ***all optional!***

<br>

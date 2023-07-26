## Upper Directory Import

<span style="color:#0000dd;">How do I import plot\_predictions from plotting.py?</span>

```
pytorch-deep-learning
├── 01_regression
│		├── 01_regression-1.py
├── toolbox
│		├── plotting.py
```

<br>

You can use the `sys` module to add the directory to the system path, which will allow you to import modules from there.

```python
import sys
sys.path.append('../toolbox')  # add directory to system path

from plotting import plot_predictions
plot_predictions(arg1, arg2)  # use the function
```

<br>

The `sys.path.append()` function adds the directory `(../toolbox)` to the list of directories that Python searches for modules.

## Import the whole file

Suppose you have a file called `my_module.py` in the same directory as your main file, and it contains the following code:

```python
def my_function():
    print("Hello from my_function!")
```

<br>
To import this module into your main file, you can use the following code:

```python
import my_module

# call the function from the imported module
my_module.my_function()
```

<br>

`import my_module` imports the `my_module.py` file into your main file.

And `my_module.my_function()` calls `my_function()` from the imported module.

## Import specific functions

If you only need to import specific functions or classes from the module, you can use the following syntax:

```python
from my_module import my_function, MyClass

# call the function from the imported module
my_function()

# create an instance of MyClass from the imported module
obj = MyClass()
```

<br>

`from my_module import my_function, MyClass` imports only the `my_function()` function and the `MyClass` class from the `my_module.py` file.

You can then call the imported functions and classes as usual.

<br>
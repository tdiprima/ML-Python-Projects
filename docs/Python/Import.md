## Upper Directory Import

Directory structure:

```
pytorch-deep-learning
├── 01_regression
│		├── 01_regression-1.py
├── toolbox
│		├── plotting.py
```

<br>
You can use the sys module to add the directory to the system path, which will allow you to import modules from there.

```python

import sys

# Import plot_predictions from plotting.py
sys.path.append('../toolbox')

from plotting import plot_predictions
plot_predictions(arg1, arg2)
```

<br>
The sys.path.append() function adds the directory (../toolbox) to the list of directories that Python searches for modules.

## Same Directory Import

Suppose you have a file called `my_module.py` in the same directory as your main file, and it contains the following code:

```python
def my_function():
    print("Hello from my_function!")
```

<br>
To import this module into your main file:

```python
import my_module

# call the function from the imported module
my_module.my_function()
```

## Cherry-pick 🍒

If you only need to import specific functions or classes from the module, you can use the following syntax:

```python
from my_module import my_function, MyClass

# call the function from the imported module
my_function()

# create an instance of MyClass from the imported module
obj = MyClass()
```

<br>

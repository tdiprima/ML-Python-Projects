## Title Of The Window

Using matplotlib, the title bar always says "Figure 1".  How do I change that?

You can change the title of a matplotlib figure window by using the `Figure.canvas.set_window_title()` method.

Here's an example of how to do it:

```python
import matplotlib.pyplot as plt

# create a plot
plt.plot([1, 2, 3], [4, 5, 6])

# set the title of the figure window
# plt.gcf().canvas.set_window_title('My Custom Title')
plt.gcf().canvas.manager.set_window_title('My Custom Title')

# display the plot
plt.show()
```

`figure = plt.gcf()`

In this example, `plt.gcf()` returns the **current figure** object, and **`canvas.manager.set_window_title()`** sets the title of the figure window to "My Custom Title".

<span style="color:red;font-size:larger;">FYI</span>

Just an FYI - `plt.gcf().canvas.set_window_title` has been replaced with `plt.gcf().canvas.manager.set_window_title`.

Thank you for letting me know! You are correct, the `Figure.canvas.set_window_title()` method has been replaced with `FigureManager.set_window_title()` since version 3.3.0 of Matplotlib.

## Title Of The Plot

Note that this method sets the title of the window itself, not the title of the plot within the window. To set the title of the plot, you can use the plt.title() method. For example:

```python
import matplotlib.pyplot as plt

# create a plot
plt.plot([1, 2, 3], [4, 5, 6])

# set the title of the plot
plt.title('My Custom Title')

# display the plot
plt.show()
```

In this example, **`plt.title()`** sets the title of the plot itself to "My Custom Title".


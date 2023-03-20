### How do I make the graph show the underlying data (X, y, and label) on hover?

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# I used sklearn's make_circles to create sample data
X, y = make_circles(1000, noise=0.03, random_state=42)

# I plotted it using matplotlib
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()
```

## Using Plotly

To make the graph show the underlying data (X, y, and label) on hover, you can use a library called **Plotly**, which provides interactive plots with hover functionality. Here's an example code that shows how to create an interactive scatter plot using Plotly:

```python
import plotly.express as px

X, y = make_circles(1000, noise=0.03, random_state=42)

fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y, hover_data={'X': X, 'y': y})
fig.show()
```

In this code, we use the `px.scatter` function from **Plotly Express** to create a scatter plot with the x and y coordinates of X, and use the color argument to color the points according to their corresponding labels y. We also pass a `hover_data` dictionary with the X and y arrays as values, which enables hover functionality with these data shown.

Finally, we use the `fig.show()` method to display the interactive plot in a new window or Jupyter notebook cell.


### It's giving me an exception that says: Data must be 1-dimensional.

I apologize for the confusion. It looks like there was a mistake in my previous answer, as the `hover_data` parameter in Plotly Express **expects a dictionary where the values are 1-dimensional arrays.**

To fix this issue, we can modify the code to create two new columns in the data frame, one for the x values and one for the y values, and then use these columns as values for the `hover_data` parameter. Here's the corrected code:

```python
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

X, y = make_circles(1000, noise=0.03, random_state=42)
df = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'label': y})

fig = px.scatter(df, x='x', y='y', color='label', hover_data={'x': df['x'], 'y': df['y'], 'label': df['label']})
fig.show()
```

In this code, we create a **Pandas DataFrame** with three columns for the x values, y values, and label values. We then pass this DataFrame to the `px.scatter` function and specify the x, y, and color parameters to create the scatter plot. Finally, we pass a dictionary to the `hover_data` parameter that includes the x, y, and label columns of the DataFrame. This will enable hover functionality with these data shown when the mouse is over a point.


### It gave me another error.  Can't the hover functionality be done using matplotlib.pyplot?

## Using mplcursors

I apologize for the confusion. Yes, it is also possible to add hover functionality to a Matplotlib scatter plot using the mplcursors library. Here's an example code that shows how to use mplcursors to display the underlying data on hover:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import mplcursors

X, y = make_circles(1000, noise=0.03, random_state=42)

fig, ax = plt.subplots()
scatter = ax.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)

cursor = mplcursors.cursor(scatter)

@cursor.connect("add")
def on_add(sel):
    index = sel.target.index
    x, y, label = X[index, 0], X[index, 1], y[index]
    sel.annotation.set_text(f"x={x:.2f}, y={y:.2f}, label={label}")

plt.show()
```

In this code, we first create a scatter plot using Matplotlib's scatter function. We then create an **mplcursors cursor** object that we can use to handle the hover events. The `cursor.connect` method takes a function that will be called when the cursor is added to a point on the scatter plot. The `sel` object that is passed to this function contains information about the selected point, including its index.

In the example code, we define the `on_add` function to display the x, y, and label values of the selected point in the hover text. We use string formatting to display these values with two decimal places. Finally, we use plt.show()` to display the plot with the hover functionality.

Note that `mplcursors` requires Matplotlib version 3.1 or higher, so you may need to upgrade your Matplotlib installation if you are using an older version.

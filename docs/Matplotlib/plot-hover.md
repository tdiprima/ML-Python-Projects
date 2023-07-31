## Hover

<span style="color:#0000dd;">How do I make the graph show the underlying data (X, y, and label) on hover?  Matplotlib shows x,y coordinates on hover.</span>

This is the sample data red and blue in circle.

X-Shape: (1000, 2)<br>
y-Shape: (1000,)

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Create sample data
X, y = make_circles(1000, noise=0.03, random_state=42)

# Plot using matplotlib
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()
```

## Using Plotly

To make the graph show the underlying data (X, y, and label) on hover, you can use a library called **Plotly**, which provides interactive plots with hover functionality. Here's an example code that shows how to create an interactive scatter plot using Plotly:

<span style="color:#0000dd;">All he did was change one line.  Instead of "plt.scatter", "px.scatter".  You need more than that.</span>

In this code, we use the `px.scatter` function from **Plotly Express** to create a scatter plot with the x and y coordinates of X.

We use the color argument to color the points according to their corresponding labels y.

We also pass a `hover_data` dictionary with the X and y arrays as values, which enables hover functionality with these data shown.

Finally, we use the `fig.show()` method to display the interactive plot.

## Plotly Hover Data Error

The error you're encountering is likely due to the format of your data that you're passing to the `hover_data` argument in `px.scatter()`. The `hover_data` argument in Plotly expects a list of column names (as strings) or a dictionary where keys represent column names and values represent either a boolean to include/exclude or a dictionary for more specific options. In your case, you're providing a dictionary with `X` and `y` keys, but the values are multi-dimensional arrays which is causing the ValueError.

If you want to include the X and y values in your hover data, you don't need to add them to `hover_data` separately as they're already the primary dimensions of your scatter plot and will be included in the hover text by default. The color dimension (which you've set as `y`) will also be included in the hover text. 

If you have additional data you want to include in the hover text, you could add that data to a Pandas DataFrame along with your `X` and `y` data, and then provide those column names in the `hover_data` argument.

```python
import plotly.express as px
import pandas as pd
from sklearn.datasets import make_circles

X, y = make_circles(1000, noise=0.03, random_state=42)

# Create a DataFrame with your X, y data
df = pd.DataFrame({
    'X0': X[:, 0],
    'X1': X[:, 1],
    'y': y
})

fig = px.scatter(df, x='X0', y='X1', color='y')
fig.show()
```

In this code, I first created a Pandas DataFrame with the X and y data. I split the two dimensions of X into two separate columns (`X0` and `X1`), which is necessary because Pandas DataFrame only accepts 1-dimensional data for each column. Then I used `px.scatter()` to create the scatter plot. The DataFrame column names are used for the x, y, and color parameters. These column values will automatically be included in the hover text.

<!-- I apologize for the confusion. It looks like there was a mistake in my previous answer, as the `hover_data` parameter in Plotly Express **expects a dictionary where the values are 1-dimensional arrays.**

To fix this issue, we can modify the code to create two new columns in the data frame, one for the x values and one for the y values, and then use these columns as values for the `hover_data` parameter. -->

## Using mplcursors

Yes, it is also possible to add hover functionality to a Matplotlib scatter plot using the `mplcursors` library.

```python
import matplotlib.pyplot as plt
import mplcursors
from sklearn.datasets import make_circles

X, y = make_circles(1000, noise=0.03, random_state=42)

fig, ax = plt.subplots()
scatter = ax.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)

cursor = mplcursors.cursor(scatter)


@cursor.connect("add")
def on_add(sel):
    index = sel.target.index
    x, y1, label = X[index, 0], X[index, 1], y[index]
    sel.annotation.set_text(f"x={x:.2f}, y={y1:.2f}, label={label}")


plt.show()
```

I click the outer red circle, everything is fine.  Click the blue circle, and I get a Warning. ⚡️

⚠️ UserWarning: Selection.target.index is deprecated and will be removed in the future; use Selection.index instead.

In this code, we first create a scatter plot using Matplotlib's scatter function. We then create an **mplcursors cursor** object that we can use to handle the hover events. The `cursor.connect` method takes a function that will be called when the cursor is added to a point on the scatter plot. The `sel` object that is passed to this function contains information about the selected point, including its index.

In the example code, we define the `on_add` function to display the x, y, and label values of the selected point in the hover text. We use string formatting to display these values with two decimal places. Finally, we use plt.show()` to display the plot with the hover functionality.

Note that `mplcursors` requires Matplotlib version 3.1 or higher, so you may need to upgrade your Matplotlib installation if you are using an older version.

<br>

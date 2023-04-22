## Pandas value\_counts() for classification

If I had classification data, and I wanted to see how many of each category there are, how can I do that using python pandas?

To count the number of each category in a Pandas dataframe column, you can use the `value_counts()` method.

```python
import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({'category': ['A', 'B', 'B', 'C', 'A', 'C', 'B']})

# Count the number of each category
counts = df['category'].value_counts()

print(counts)
```

This will output the following:

```css
B    3
A    2
C    2
Name: category, dtype: int64
```

In this example, there are three categories 'B', two categories 'A', and two categories 'C' in the 'category' column of the dataframe. The output shows the count of each category and their respective values.

You can create a Pandas DataFrame from an array of training data and an array of labels by using the pd.DataFrame function.

Here's an example:

```python
import pandas as pd
import numpy as np

# Create sample training data and labels arrays
train_data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
labels = np.array([0, 1, 0])

# Create a Pandas DataFrame
df = pd.DataFrame(data=train_data, columns=['feature_1', 'feature_2', 'feature_3'])
df['label'] = labels

print(df)
```

In this example, we create a Pandas DataFrame `df` from the `train_data` array and `labels` array.

We first create a DataFrame using the `pd.DataFrame` function and pass in the training data `train_data`.

We also specify the column names of the DataFrame using the `columns` parameter.

Finally, we add a new column to the DataFrame called `label` and assign the labels array to it.

The output of the above code will be:

```css
   feature_1  feature_2  feature_3  label
0          0          1          2      0
1          3          4          5      1
2          6          7          8      0
```

Here, you can see that the DataFrame df contains the training data as well as the corresponding labels in a separate column.

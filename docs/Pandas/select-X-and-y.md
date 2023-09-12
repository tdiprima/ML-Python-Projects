## Selecting Pandas DataFrame Columns

If I have a pandas dataframe, and select the feature data like:

```py
X = dataset.iloc[:, :-1].values
```

<br>
this selects everything except the last column.  The values are returned as a NumPy array.

<br>
Then, if I want to select the last column and put it in a variable y, it's -1 without the : in front of it.

```py
y = dataset.iloc[:, -1].values
```

<br>
Here's a quick example for better understanding:

```python
import pandas as pd
import numpy as np

# Create a simple DataFrame for demonstration
dataset = pd.DataFrame({
    'Feature1': [1, 2, 3],
    'Feature2': [4, 5, 6],
    'Target': [7, 8, 9]
})

# Select all but the last column
X = dataset.iloc[:, :-1].values

# Select only the last column
y = dataset.iloc[:, -1].values

print("X:", X)
print("y:", y)
```

<br>
Output:

```
X: [[1 4]
    [2 5]
    [3 6]]
y: [7 8 9]
```

<br>

So in this example, `X` will be a 2D NumPy array containing the 'Feature1' and 'Feature2' data, while `y` will be a 1D NumPy array containing the 'Target' data.

## Anatomy of the darn thing

```py
X = dataset.iloc[:, :-1].values
```

<br>

`iloc` is used to select a subset of the dataframe `dataset`.

The `[:, :-1]` notation specifies a range of rows and columns to select.

The **first `:`** indicates that we want to select all rows of the dataset.

The **second `:`** indicates that we want to select all columns.

The **`-1`** indicates that we want to select all columns except for the last one.

So `[:, :-1]` selects all rows of the dataset and all columns except for the last column.

The resulting `X` will be a numpy array containing all the rows of `dataset` and all the columns except for the last one.

Note that the `iloc` method selects data by **position**, not by column name.

If you have **column names**, you can use the `loc` method to select data by name, e.g. `dataset.loc[:, 'column_name']`.

## Reverse a sequence

```py
[::-1] # no comma
```

<br>

The `[::-1]` syntax is used to reverse a sequence, such as a list, tuple, or NumPy array.

```sh
# ACK, by the way
ack "\[\:, \:\-1\]"
```

<br>
## See also

`[::-1]` [SupervisedLearn1.ipynb](../../Keras/SupervisedLearn1.ipynb)

`[:, :-1]` [heart\_disease\_detection.py](../../Scikit-learn/everything\_else/heart\_disease\_detection.py)

<br>

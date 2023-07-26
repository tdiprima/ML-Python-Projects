## Selecting Data

I'm reading a dataset from a csv to pandas dataframe.

```py
# Select the feature data
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

<br>

```py
# Pop the target data
y = dataset.iloc[:, 4].values
```

<br>

The `iloc` method is again used to select a subset of the dataframe. This time, `[:, 4]` **selects all rows** of the dataset **and the 4th column** (index starting at 0). 

This suggests that the dataset has 5 columns, and the **target variable** (the variable to be predicted) is stored in the 5th column (index 4). The resulting `y` will be a numpy array containing the target variable.

Note that the `iloc` method selects data by **position**, not by column name.

If you have **column names**, you can use the `loc` method to select data by name, e.g. `dataset.loc[:, 'column_name']`.


## Reverse a sequence

```py
[::-1] # no comma
```

The `[::-1]` syntax is used to reverse a sequence, such as a list, tuple, or NumPy array.

```sh
# ACK, by the way
ack "\[\:, \:\-1\]"
```

## See also

`[::-1]` [SupervisedLearn1.ipynb](../../Keras/SupervisedLearn1.ipynb)

`[:, :-1]` [heart\_disease\_detection.py](../../Scikit-learn/everything\_else/heart\_disease\_detection.py)

<br>

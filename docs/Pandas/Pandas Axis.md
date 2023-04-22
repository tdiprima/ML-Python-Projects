# Counting Null Values

## axis=0 and axis=1

In Pandas dataframe, if I do: `dataframe.isnull().sum(axis=0)`, does the `axis=0` mean to check each row?  And, does it stand to reason that I could have done axis=1, and the effect would still be the same?

No, the `axis=0` parameter in the `dataframe.isnull().sum(axis=0)` method call means that you want to **calculate the sum of null values for each column** of the dataframe, not each row.

To check for null values in each row of the dataframe, you would use `axis=1` instead, like this: `dataframe.isnull().sum(axis=1)`.

This would return a **pandas series** with the count of null values for each row.

So, if you had used `axis=1` instead of `axis=0` in `dataframe.isnull().sum(axis=0)`, you would get a different result.

The axis parameter specifies the axis along which the operation should be performed, so changing it would change the way the null values are counted.

## Okay, prove it.

Let's say I forget which is which.  Is there a way, using pandas, to display "axis=0"?

Yes, you can display the index value of the axis by calling the `index` attribute of the respective axis.

For example, to display the index value of `axis=0`, you can use the following code:

```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, None], 'C': [7, 8, 9]})

print(df.isnull().sum(axis=0).index)

# Index(['A', 'B', 'C'], dtype='object')
```

The `Index` object contains the column names of the dataframe, which confirms that `axis=0` refers to the "columns" axis.

## axis=1 and RangeIndex

And `df2.isnull().sum(axis=1).index` renders `RangeIndex(start=0, stop=3, step=1)`.  What the heck is RangeIndex?  Also - I'm guessing that when we do this, the output says what the possible values could be, not what they actually are?  Because in the first example, column C does not have any NaN in it.

`RangeIndex` is a class in pandas that represents a range of consecutive integers as an index.

In this case, `RangeIndex(start=0, stop=3, step=1)` indicates that the index for the resulting series has three elements, with labels starting at 0 and ending at 2 (inclusive), with a step of 1.

You are correct that the output of `df.isnull().sum(axis=0).index` and `df2.isnull().sum(axis=1).index` shows the **possible** column or row labels, respectively, regardless of whether they actually contain NaN values.

In the first example, column C does not have any NaN values, but it is still listed in the index output because it is a valid column label in the dataframe.

Note that the `isnull().sum()` operation is counting the number of missing values (NaN) in each column or row of the dataframe, regardless of whether the other values in that column or row are present or not.

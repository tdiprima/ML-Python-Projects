## Create dataframe

```ruby
import pandas as pd

df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, None], 'C': [7, 8, 9]})
```

<br>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

<br>

## Axes

`axis=0` is column

`axis=1` is row

<br>

## Counting Null Values

Calculate the sum of null values for

### Each column of the dataframe

```ruby
df.isnull().sum(axis=0)
```

<br>

```c
A    1
B    2
C    0
dtype: int64
```

<br>
<span style="font-weight:bold">Meaning:</span><br>
A - 1 is null<br>
B - 2 are null<br>
C - none are null<br>

Note that the `isnull().sum()` operation is counting the number of missing values (NaN) in each column or row of the dataframe.

<br>

## Check for null values in each row

You would use `axis=1` instead.

This would return a **pandas.Series** with the count of null values for each row.


```ruby
result = df.isnull().sum(axis=1)
```

<br>

## How to check which is which

You can display the index value of the axis by calling the `index` attribute of the respective axis.

For example, to display the index value of `axis=0`, you can use the following code:


```ruby
df.isnull().sum(axis=0).index
```

<br>

```c
Index(['A', 'B', 'C'], dtype='object')
```

The `Index` object contains the column names of the dataframe, which confirms that `axis=0` refers to the "columns" axis.

### Let's try it for the row


```python
df.isnull().sum(axis=1).index
```

<br>

```c
RangeIndex(start=0, stop=3, step=1)
```

<br>

## See: Index and RangeIndex

`RangeIndex` is a class in pandas that represents a range of consecutive integers as an index.

In this case, `RangeIndex(start=0, stop=3, step=1)` indicates that the index for the resulting series has three elements, with labels starting at 0 and ending at 2 (inclusive), with a step of 1.

<br>

## value\_counts() for classification

<span style="color:#0000dd;">I have classification data, and I want to see how many of each category there are.</span>

To count the number of each category in a Pandas dataframe column, you can use the `value_counts()` method.

```python
import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({'category': ['A', 'B', 'B', 'C', 'A', 'C', 'B']})

# Count the number of each category
counts = df['category'].value_counts()

print(counts)
```

<br>
This will output the following:

```css
B    3
A    2
C    2
Name: category, dtype: int64
```

<br>
In this example, there are three categories 'B', two categories 'A', and two categories 'C' in the 'category' column of the dataframe. The output shows the count of each category and their respective values.

<br>

## Training Data to DataFrame

You can create a Pandas DataFrame from an array of training data and an array of labels by using the `pd.DataFrame` function.

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

<br>
The output of the above code will be:

```css
   feature_1  feature_2  feature_3  label
0          0          1          2      0
1          3          4          5      1
2          6          7          8      0
```

<br>
You can see that the DataFrame `df` contains the training data as well as the corresponding labels in a separate column.

<br>

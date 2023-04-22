## Clara

How can I query the dataframe for all rows where part of the name has "Clara" in it?

You can use the `str.contains()` method in Pandas to check whether a string contains a specific substring, and then use this method to filter the rows based on whether the "full name" column contains the string "Clara".

``` python
import pandas as pd

# create a sample dataframe
df = pd.DataFrame({
    'full name': ['Clara Smith', 'John Clarke', 'Mary Clarke', 'Robert Clark']
})

# filter the dataframe based on whether the "full name" column contains "Clara"
clara_df = df[df['full name'].str.contains('Clara')]

print(clara_df)
```

This will output the following filtered dataframe:

```
    full name
0  Clara Smith
```

In this example, only the row with the name "Clara Smith" contains the string "Clara" in the "full name" column, so this is the only row that is returned in the filtered dataframe.

```py
df = train_data
# Beatrice Wood
# Rose Dewitt Bukater
rose_df = df[df['Name'].str.contains('Bea')]
rose_df  # oh well.
```

## Women

I wanted to try to figure out what percentage of women died.  Rather than subtract 100% - survived %.

```py
w = train_data.loc[train_data.Sex == 'female']["Survived" == 0]
```

The error you're encountering is because the **syntax for subsetting** a DataFrame in pandas is incorrect in the code you wrote.

Specifically, you're using the equality operator `==` **inside the indexing brackets** (which is wrong).

Here's how you can **fix the code** to calculate the percentage of female passengers who did not survive:

```python
# Subset the data to female passengers
female_data = train_data[train_data['Sex'] == 'female']

# Count the number of females who did not survive
num_female_died = len(female_data[female_data['Survived'] == 0])

# Calculate the percentage of females who did not survive
percent_female_died = num_female_died / len(female_data) * 100

print(f"Percentage of female passengers who did not survive: {percent_female_died:.2f}%")
```

<mark>**TODO:**</mark> Compare with existing Kaggle code for survival.

<mark>**AND:**</mark> Compare with Tim's code.

1. Subset the data to only include female passengers
    * Create a boolean mask that filters the DataFrame based on the condition that the 'Sex' column equals 'female'.

2. Count the number of female passengers who did not survive
    * Create another boolean mask that filters the `female_data` DataFrame based on the condition that the 'Survived' column equals 0
    * Calculate the length of the resulting DataFrame.

3. Calculate the percentage of female passengers who did not survive
    * Divide the number of females who did not survive by the total number of females in the dataset
    * Multiply by 100.

4. Print result with **two decimal places** using f-strings.

<br>

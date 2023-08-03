## string.contains()

How can I query the dataframe for all rows where part of the name has "Alejandra" in it?

You can use the `str.contains()` method in Pandas to check whether a string contains a specific substring, and then use this method to filter the rows based on whether the "full name" column contains the string "Alejandra".

``` python
import pandas as pd

# create a sample dataframe
df = pd.DataFrame({
    "full name": [
        "Daniella Villarreal",
        "Paulina Villarreal",
        "Alejandra Villarreal"
    ]})

# filter the dataframe based on whether the "full name" column contains "Alejandra"
series = df["full name"].str.contains("Alejandra")
df1 = df[series]
```

<br>
This will output the following filtered dataframe:

```
              full name
2  Alejandra Villarreal
```

<br>
And Series gives you:

```
0    False
1    False
2     True
Name: full name, dtype: bool
```

## Filter Names: Beatrice & Rose

```py
# Looking for "Rose Dewitt Bukater" aka "Beatrice Wood"
df = train_data
rose_df = df[df['Name'].str.contains('Beatrice|Rose')]
rose_df  # Leah Rosen. Eh, wrong Rose. But she survived. :)
```

<br>

You can filter a pandas dataframe for those rows whose 'Name' string contains either "Beatrice" or "Rose" by using the pipe operator `|` to represent the logical OR.

```python
df[df['Name'].str.contains('Beatrice|Rose')]
```

<br>

Note that the function `str.contains()` returns a boolean Series with the same index as the original Series and this Series is used to filter the dataframe. The pipe operator `|` is a regex OR, matching a pattern either before or after the pipe.

Also, be aware that `str.contains()` is case sensitive by default. If you want to make it case insensitive, use the `case` parameter and set it to `False`:

```python
df[df['Name'].str.contains('Beatrice|Rose', case=False)]
```

<br>
This will match "Beatrice", "BEATRICE", "beatrice", "Rose", "ROSE", "rose", etc.

## Titanic 🚢

<span style="color:#0000dd;">I wanted to try to figure out what percentage of women died.  Rather than subtract 100% - survived %.</span>

[Kaggle.com](https://www.kaggle.com/code/tammydiprima/coherent-firebird)

Calculate the percentage of female passengers who did not survive:

```python
# Subset the data to female passengers
f_series = train_data['Sex'] == 'female'  # True and False
female_data = train_data[f_series]  # dataframe of only women

# Count the number of females who did not survive
fs_series = female_data['Survived'] == 0
fs_df = female_data[fs_series]
num_female_died = len(fs_df)

# Calculate the percentage of females who did not survive
percent_female_died = num_female_died / len(female_data) * 100

print(f"Percentage of female passengers who did not survive: {percent_female_died:.2f}%")
# 25.80%
```

<br>

1. Subset the data to only include female passengers
    * Create a boolean mask that filters the DataFrame based on the condition that the 'Sex' column equals 'female'.

2. Count the number of female passengers who did not survive
    * Create another boolean mask that filters the `female_data` DataFrame based on the condition that the 'Survived' column equals 0
    * Calculate the length of the resulting DataFrame.

3. Calculate the percentage of female passengers who did not survive
    * Divide the number of females who did not survive by the total number of females in the dataset
    * Multiply by 100.

4. Print result with two decimal places using f-strings.

## Survived

```py
survived_women = train_data.loc[train_data.Sex == 'female']["Survived"]
pct_women_survived = sum(survived_women) / len(survived_women) * 100
# 233 / 314 * 100

print("\n% of women who survived:", pct_women_survived)
# 74.20382165605095
```

## YouTube

I wanted to find these people from a story I watched on YouTube:

**Found**

* Countess of ~~Rhodes~~ Rothes
* Señora de Satode Peñasco
    * Of course back in the day they typed 'n' instead of 'ñ'
* Joseph Philippe Lemercier Laroche &ndash; engineer
* Juliette LaFargue
* Simonne and Louise

**Not found**

* Violet Constance Jessop
* Joseph Laroche
* Charles Joughin
* Louis Lolo
* Dorothy Gibson

```py
series1 = train_df["Name"].str.contains("Rothes")
series2 = train_df["Name"].str.contains("de Satode")
series3 = train_df["Name"].str.contains("Laroche")

# Combine series using bitwise OR
combined_series = series1 | series2 | series3

# Create a new DataFrame with the rows where any of the series are True
df1 = train_df[combined_series]

# Sort by the "Name" colum
df1 = df1.sort_values(by='Name')  # Ascending
# df1 = df1.sort_values(by='Name', ascending=False)  # Sort in descending order
df1
```

[kaggle](https://www.kaggle.com/code/tammydiprima/coherent-firebird/edit)

Please note that `sort_values()` returns a new DataFrame and does not modify the original one in-place. If you want to modify the original DataFrame, you can use the `inplace=True` argument:

```python
df1.sort_values(by='Name', inplace=True)
```

The `inplace=True` argument will cause the method to modify the DataFrame in place (do not create a new object). Changes are reflected in the original DataFrame.

<br>

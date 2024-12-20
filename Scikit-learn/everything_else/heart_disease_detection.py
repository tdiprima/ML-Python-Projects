"""
Reads data from a cardiovascular dataset, explores and cleans it, and visualizes the distribution of heart disease
across different ages. The program checks correlations among features, drops unnecessary columns, and normalizes the data.
Finally, it uses a Random Forest Classifier to predict whether someone has heart disease, evaluating the model's accuracy
on training data and validating it using a confusion matrix on test data.

https://youtu.be/kySc5Wg1Gxw
https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118
"""

import pandas as pd
import seaborn as sns

# Toy dataset with 7 rows
df = pd.read_csv("cardio.csv")
print("\ndf:\n", df)

rows, columns = df.shape
print(f"\nThere are {rows} rows and {columns} columns.")

# Count the empty of null values in each column
count_null = df.isna().sum()
print("\ncount_null:\n", count_null)

# Is there any missing data
is_missing = df.isnull().values.any()
print("\nis_missing:\n", is_missing)

# View some basic statistics
print("\nDescribe:\n", df.describe())

# Days to years
# round(19514 / 365)

# Max age
# round(22113 / 365)

# 0 = no heart disease, 1 = with heart disease
print("\nValue counts:\n", df["cardio"].value_counts())

import warnings

# plot without specifying the x, y parameters
warnings.simplefilter(action="ignore", category=FutureWarning)

# Visualize the count
from matplotlib import pyplot as plt

sns.countplot(df["cardio"])
plt.show()

# x = data['Healthy life expectancy']
# y = data['max_dead']
# sns.regplot(x=x, y=y)


# Create a years column
df["years"] = round(df["age"] / 365)
df["years"] = pd.to_numeric(df["years"], downcast="integer")
print("\nAge in Years:\n", df["years"])

# Visualize
sns.countplot(
    x="years",
    hue="cardio",
    data=df, palette="colorblind",
    edgecolor=sns.color_palette("dark", n_colors=1)
)
plt.show()

# Remember we only have 7 rows in our data
# If we had more rows, we could easily see the ages where cv disease exceeds no cv disease.


# See how the columns are correlated with each other
# Remember?  It's a confusion matrix.  "id" is 100% correlated with "id".
# Since id doesn't matter, maybe we wanna drop the column before training our model.
print("\nCorrelation:\n", df.corr())

# Visualize (x3)
plt.figure(figsize=(7, 7))
# sns.heatmap(df.corr, annot=True, fmt=".0%")  # Percentage w/o decimal places
# Error: Must pass 2-d input. shape=() - simply means you forgot the "()"!
sns.heatmap(df.corr(), annot=True, fmt=".0%")
plt.show()

# Cardio &ndash; 30% positive correlation with age.
#
# And then look at height and weight, etc. &ndash; things you'd expect to see.
#
# Notice &ndash; years and age have the same correlation percentage, because they're the same thing.
#
# And they have 100% correlation with each other.  Voila.
#
# We needed the years column for something else, but we can get rid of it for training.

df = df.drop("years", axis=1)  # 1 = column
df = df.drop("id", axis=1)
# df.head()


# Split data into features and targets
# X = independent data; y = target data

X = df.iloc[:, :-1].values  # All of the rows, and all columns except the last column

y = df.iloc[:, -1].values  # Retain the last column for target data

# Split data into training and test (75% / 25%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# len(X_train), len(X_test), len(y_train), len(y_test)


# Feature scaling
# Which we know and love as normalization
# Do it between 0 and 1
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Use RandomForest classifier
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=1)

forest.fit(X_train, y_train)  # Train model

# Test the model's accuracy on the training dataset
model = forest
score = model.score(X_train, y_train)
print("\nScore:\n", score)

# Test the model's accuracy on the test dataset
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

true_negative = cm[0][0]
true_positive = cm[1][1]

false_negative = cm[1][0]
false_positive = cm[0][1]

# Print confusion matrix
print(f"\nConfusion matrix:\n{cm}")

# Formula for accuracy: (TP + TN) / (all the values)
acc = (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)

# Print the model's accuracy on the test data
print(f"\nModel Test Accuracy:\n{acc}")

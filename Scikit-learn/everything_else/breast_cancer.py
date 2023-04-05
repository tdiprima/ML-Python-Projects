#!/usr/bin/env python
# coding: utf-8

# FYI: Random Forest ended up working better than KNeighborsClassifier.

import random
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # Try decision tree, random forest

# To ignore warning at clf.score(X_test, y_test)
# https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
simplefilter(action='ignore', category=FutureWarning)

# Get Data
data = load_breast_cancer()

# Describe data
print("\nKeys:\n", data.keys())
print("\nFeatures:\n", data["feature_names"])  # X values
print("\nTargets:\n", data["target_names"])  # y values

# Split data
X = data['data']
y = data['target']

# Bincount
counts = np.bincount(y)
# counts


# According to [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data):
#
# Class distribution: 357 benign, 212 malignant
#
# Diagnosis (M = malignant, B = benign) <span style="color:red;">&lt;&ndash; !</span>

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# len(X_train), len(X_test), len(y_train), len(y_test)


# Classify
# Using KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

print("\nScore:\n", clf.score(X_test, y_test))

# Predict on real data
#
# Let's say we have a spreadsheet of features.
#
# `X_new = np.array([3.4, 1.2...])` <- 30 values
#
# Feed them to the predict function.
#
# Today we're just gonna fabricate them.  Just one row.

num_features = len(data["feature_names"])
# num_features


# Create a random sample; give it a list, and how many.
# Throw it into a numpy array.
X_new = np.array(random.sample(range(0, 50), num_features))
# X_new


# Make a prediction
what_class = clf.predict([X_new])  # Expects a 2D array, hence the brackets.
# what_class


what_name = data["target_names"][what_class[0]]
print(what_name)

# Which is which
# That answers your question on which is which.

benign = 1
malignant = 0

# That's all you do!

label_1 = np.array([0])
label_2 = np.array([1])

name_1 = data['target_names'][label_1[0]]
name_2 = data['target_names'][label_2[0]]

print(f"{label_1[0]} is {name_1} and {label_2[0]} is {name_2}")

# Pandas DataFrame
#
# https://youtu.be/ULvlqwjNNAo

# Concatenate...
column_data = np.concatenate([data["data"], data["target"][:, None]], axis=1)
column_names = np.concatenate([data["feature_names"], ["class"]])

# column_data.shape

# column_names is just the feature names, from above, plus "class"

# len(column_names)

df = pd.DataFrame(column_data, columns=column_names)
# df

# Correlations

# Oh, look - it's a confusion matrix.
correlations = df.corr()
# correlations

# Plot correlations

# sns.heatmap(correlations, cmap="coolwarm", annot=True, annot_kws={"fontsize": 8})
# plt.tight_layout()
# plt.show()

sns.heatmap(correlations, cmap="RdBu")
plt.show()

# Color Palette

# Make it fail, and it tells you all the colors it should be.
try:
    sns.heatmap(correlations, cmap="dog")
except ValueError as ve:
    print(ve)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier()

# Fit on the train
clf1.fit(X_train, y_train)

# Score on the test
score1 = clf1.score(X_test, y_test)
print(score1)

pred = clf1.predict([X_new])
# data["target_names"][pred[0]]


# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier()

# Fit on the train
clf2.fit(X_train, y_train)

# Score on the test
score2 = clf2.score(X_test, y_test)
print(score2)

pred2 = clf2.predict([X_new])
data["target_names"][pred2[0]]

# Which classifier had the highest score?

classifiers = ['KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier']
scores = [0.9035087719298246, 0.9035087719298246, 0.9298245614035088]

# Our friend, argmax.
mmax = np.argmax(scores)

print(classifiers[mmax])

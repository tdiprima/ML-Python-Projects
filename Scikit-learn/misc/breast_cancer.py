# https://youtu.be/ULvlqwjNNAo
import random
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # decision tree, random forest

# To ignore warning at clf.score(X_test, y_test)
# https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
simplefilter(action='ignore', category=FutureWarning)

# GET DATA
data = load_breast_cancer()

# print("\nKeys", data.keys())
# print("\nFeatures", data["feature_names"])  # X values
# print("\nTargets", data["target_names"])  # y values

X = data['data']
y = data['target']

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print("Data len:", len(X_train), len(X_test), len(y_train), len(y_test))

# CLASSIFY
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

print("\nScore:", clf.score(X_test, y_test))

"""
So like, let's say we have a spreadsheet of features.
X_new = np.array([3.4, 1.2...]) <- 30 values
Feed them to the predict function.
So we're just gonna make them up.
"""
num_features = len(data["feature_names"])
# print("\nNum features:", num_features)  # 30
X_new = np.array(random.sample(range(0, 50), num_features))

what_class = clf.predict([X_new])  # Expects a 2D array
print(what_class)

what_name = data["target_names"][what_class[0]]
print(what_name)

# PANDAS
column_data = np.concatenate([data["data"], data["target"][:, None]], axis=1)
column_names = np.concatenate([data["feature_names"], ["class"]])

# DATA FRAME
df = pd.DataFrame(column_data, columns=column_names)
# print("\nData Frame:\n", df)

# CORRELATIONS
correlations = df.corr()
# print("\nCorrelations:\n", correlations)

sns.heatmap(correlations, cmap="coolwarm", annot=True, annot_kws={"fontsize": 8})
# plt.tight_layout()
plt.show()

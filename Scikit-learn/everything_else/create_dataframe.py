import pandas as pd
from sklearn.datasets import load_breast_cancer


def one_column(column_index):
    """
    column_index = Index of the column you want to extract
    """
    # Load the breast cancer dataset
    data = load_breast_cancer()

    # Get the features (X) and target (y) data
    X, y = data['data'], data['target']

    # Create a DataFrame from the nth column of X
    column_name = data['feature_names'][column_index]  # Name of the column

    # nth column of X = X[:, column_index]
    df = pd.DataFrame(X[:, column_index], columns=[column_name])

    # The resulting DataFrame will have one column with the name of the extracted feature.
    print("\nHead:\n", df.head())

    print("\nDescribe:\n", df.describe())


def all_columns():
    """
    Create a pandas DataFrame from all the columns by passing
    the entire X_train array to the pd.DataFrame() constructor.
    NOW - When I did it previously, it was wrong. But - different dtype.
    """
    # Load the breast cancer dataset
    data = load_breast_cancer()

    # Get the features (X) and target (y) data
    X, y = data['data'], data['target']

    # Create a DataFrame from all columns of X
    df = pd.DataFrame(X, columns=data['feature_names'])

    print("\nHead:\n", df.head())

    print("\nDescribe:\n", df.describe())


def get_info():
    import numpy as np
    from sklearn.model_selection import train_test_split

    # First, I want data.
    data = load_breast_cancer()

    # Now, I want it separated.
    X, y = data.data, data.target

    # I want a train/test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # What are the distinct/unique class names?
    class_names = np.unique(y_train)
    print("\nclass_names:", class_names)

    # Talk to me about the data.
    print("\nkeys:\n", data.keys())

    print("\nClass names:\n", data["target_names"])

    print("\nFeature names:\n", data["feature_names"])

    # Stick it in a data frame.
    df = pd.DataFrame(data=y_train)

    # Tally up the classes.
    counts = df.value_counts()
    print("\nValue counts:\n", counts)


if __name__ == '__main__':
    # one_column(0)
    # all_columns()
    get_info()

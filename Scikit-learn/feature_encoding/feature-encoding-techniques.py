#!/usr/bin/env python3

"""Because fireside chat.
https://towardsdatascience.com/feature-encoding-techniques-in-machine-learning-with-python-implementation-dbf933e64aa

The encoder must be fitted on only the training data, such that the encoder
only learns the categories that exist in the training set, and then be used
to transform the validation/test data. Do not fit the encoder on the whole dataset!

Save your Encoders

https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
https://contrib.scikit-learn.org/category_encoders/
"""

import pandas as pd

# TRAINING DATA
df1 = pd.DataFrame({'animal': ["aardvark", "antelope", "bass", "chicken", "clam", "flea", "frog"]})
df2 = pd.DataFrame({'type': ["mammal", "mammal", "fish", "bird", "invertebrate", "insect", "amphibian"]})

data_train = pd.concat([df1, df2], axis=1)
# print("\ndata_train\n", data_train)

# TEST DATA
T = pd.DataFrame({'animal': ["butterfly", "frog", "sparrow", "salmon", "ant", "jellyfish", "elephant"]})
D = pd.DataFrame({'type': ['insect', 'amphibian', 'bird', 'fish', 'insect', 'invertebrate', 'mammal']})

data_test = pd.concat([T, D], axis=1)


# print("\ndata_test\n", data_test)


def label_encoder():
    """
    Used for nominal categorical variables (categories without order i.e., red, green, blue)
    """
    from sklearn.preprocessing import LabelEncoder

    # Initialize Label Encoder
    encoder = LabelEncoder()

    # Fit encoder on training data
    data_train["type_encoded"] = encoder.fit_transform(data_train["type"])
    # print(data_train["type_encoded"])

    # Transform test data
    data_test["type_encoded"] = encoder.transform(data_test["type"])
    print("\ndata_test:\n", data_test)

    # Retrieve the categories (returns list)
    print("\nClasses:", list(encoder.classes_))

    # Retrieve original values from encoded values
    data_train["type2"] = encoder.inverse_transform(data_train["type_encoded"])
    print("\ndata_train:\n", data_train)


def ordinal_encoder():
    """
    TODO: Create data.
    Used for ordinal categorical variables (categories with order i.e., small, medium, large)
    It can encode multiple columns at once, and the order of the categories can be specified.
    """
    from sklearn.preprocessing import OrdinalEncoder

    # Initialize Ordinal Encoder
    encoder = OrdinalEncoder(
        categories=[["small", "medium", "large"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    data_train["size_encoded"] = encoder.fit_transform(data_train[["size"]])
    data_test["size_encoded"] = encoder.transform(data_test[["size"]])

    # Retrieve the categories (returns list of lists)
    print("\nCategories:", encoder.categories)

    # Retrieve original values from encoded values
    data_train["size2"] = encoder.inverse_transform(data_train[["size_encoded"]])
    print("\ndata_train\n:", data_train)


def one_hot(data_train=None, data_test=None):
    from sklearn.preprocessing import OneHotEncoder

    # Initialize One-Hot Encoder
    encoder = OneHotEncoder(handle_unknown="ignore")

    # Fit encoder on training data (returns a separate DataFrame)
    data_ohe = pd.DataFrame(encoder.fit_transform(data_train[["type"]]).toarray())
    print("\ndata_ohe:\n", data_ohe)
    data_ohe.columns = [col for cols in encoder.categories_ for col in cols]
    print("\ndata_ohe.columns:\n", data_ohe.columns)

    # Join encoded data with original training data
    data_train = pd.concat([data_train, data_ohe], axis=1)
    print("\ndata_train:\n", data_train)

    # Transform test data
    data_ohe = pd.DataFrame(encoder.transform(data_test[["type"]]).toarray())
    data_ohe.columns = [col for cols in encoder.categories_ for col in cols]
    data_test = pd.concat([data_test, data_ohe], axis=1)
    print("\ndata_test:\n", data_test)


def hey_dummies(data_train=None):
    data_ohe = pd.get_dummies(data_train["type"])
    print("\ndata_ohe:\n", data_ohe)

    data_train = pd.concat([data_train, data_ohe], axis=1)
    print("\ndata_train:\n", data_train)


def target_encoding():
    """
    pip install category_encoders
    TODO: Wth is "label"?
    """
    import category_encoders as ce

    # Target (Mean) Encoding - fit on training data, transform test data
    encoder = ce.TargetEncoder(cols="type", smoothing=1.0)
    data_train["type_encoded"] = encoder.fit_transform(data_train["type"], data_train["label"])
    data_test["type_encoded"] = encoder.transform(data_test["type"], data_test["label"])

    # Leave One Out Encoding
    encoder = ce.LeaveOneOutEncoder(cols="type")
    data_train["type_encoded"] = encoder.fit_transform(data_train["type"], data_train["label"])
    data_test["type_encoded"] = encoder.transform(data_test["type"], data_test["label"])

    print("\ndata_train:\n", data_train)
    print("\ndata_test:\n", data_test)


def count_freq_encoding():
    import category_encoders as ce

    # Count Encoding - fit on training data, transform test data
    encoder = ce.CountEncoder(cols="type")
    data_train["type_count_encoded"] = encoder.fit_transform(data_train["type"])
    data_test["type_count_encoded"] = encoder.transform(data_test["type"])

    # Frequency (normalized count) Encoding
    encoder = ce.CountEncoder(cols="type", normalize=True)
    data_train["type_frequency_encoded"] = encoder.fit_transform(data_train["type"])
    data_test["type_frequency_encoded"] = encoder.transform(data_test["type"])

    print("\ndata_train:\n", data_train)
    print("\ndata_test:\n", data_test)


def binary_encoding(data_train=None, data_test=None):
    import category_encoders as ce

    # Binary Encoding - fit on training data, transform test data
    encoder = ce.BinaryEncoder()
    data_encoded = encoder.fit_transform(data_train["type"])
    data_train = pd.concat([data_train, data_encoded], axis=1)

    data_encoded = encoder.transform(data_test["type"])
    data_test = pd.concat([data_test, data_encoded], axis=1)

    # BaseN Encoding - fit on training data, transform test data
    encoder = ce.BaseNEncoder(base=5)
    data_encoded = encoder.fit_transform(data_train["type"])
    data_train = pd.concat([data_train, data_encoded], axis=1)
    print("\ndata_train\n", data_train)

    data_encoded = encoder.transform(data_test["type"])
    data_test = pd.concat([data_test, data_encoded], axis=1)
    print("\ndata_test\n", data_test)


def hash_encoding_1(data_train=None, data_test=None):
    """
    todo: data_train is right, but getting NaN in data_test
    """
    from sklearn.feature_extraction import FeatureHasher

    # Hash Encoding - fit on training data, transform test data
    encoder = FeatureHasher(n_features=2, input_type="string")
    data_encoded = pd.DataFrame(encoder.fit_transform(data_train["type"]).toarray())
    print("\ndata_encoded\n", data_encoded)
    data_train = pd.concat([data_train, data_encoded], axis=1)
    print("\ndata_train\n", data_train)

    data_encoded = pd.DataFrame(encoder.transform(data_test).toarray())
    print("\ndata_encoded\n", data_encoded)

    data_test = pd.concat([data_test, data_encoded], axis=1)
    print("\ndata_test\n", data_test)


def hash_encoding_2(data_train=None, data_test=None):
    """
    NOTE: RIDICULOUSLY SLOW!
    """
    import category_encoders as ce

    # Hash Encoding - fit on training data, transform test data
    encoder = ce.HashingEncoder(n_components=2)
    data_encoded = encoder.fit_transform(data_train["type"])
    data_train = pd.concat([data_train, data_encoded], axis=1)
    print("\ndata_train\n", data_train)

    data_encoded = encoder.transform(data_test["type"])
    data_test = pd.concat([data_test, data_encoded], axis=1)
    print("\ndata_test\n", data_test)


if __name__ == '__main__':
    label_encoder()
    # one_hot(data_train, data_test)
    # hey_dummies(data_train)
    # target_encoding()
    # count_freq_encoding()
    # binary_encoding(data_train, data_test)
    # hash_encoding_1(data_train, data_test)
    # hash_encoding_2(data_train, data_test)

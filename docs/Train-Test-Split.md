## Train test split

<span style="color:#0000dd;">In what order does sklearn.model\_selection's train\_test\_split return values?</span>

`sklearn.model_selection`'s `train_test_split` function returns the values in the following order:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<br>

1. **`X_train`**: This variable contains the training data features (input variables) that will be used to train your machine learning model. It is a subset of the original `X` data.

2. **`X_test`**: This variable contains the testing data features (input variables) that will be used to evaluate the performance of your trained model. It is also a subset of the original `X` data.

3. **`y_train`**: This variable contains the corresponding target values (labels) for the training data. It is a subset of the original `y` data.

4. **`y_test`**: This variable contains the corresponding target values (labels) for the testing data. It is also a subset of the original `y` data.

<br>

The `train_test_split` function is commonly used to split a dataset into training and testing sets. The `test_size` parameter determines the proportion of the data that will be used for testing (e.g., `test_size=0.2` means 20% of the data will be used for testing). The `random_state` parameter is optional and is used to set a random seed for reproducibility.

<br>

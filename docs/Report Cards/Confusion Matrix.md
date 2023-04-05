## Confusion Matrix

A confusion matrix is a way to see **how well a machine learning model** is doing at **classifying things correctly.**

### Code

```ruby
confusion_matrix(truth, prediction)
```

### Missing an argument

"TypeError: `__new__()` missing 1 required positional argument: 'task'

This error occurs when you're trying to create an instance of the `ConfusionMatrix` class without providing the required argument `task`.

<span style="color:#0000dd;font-size:larger;">Duh. 🤦‍♀️</span>

So it goes like this:

```py
from confusion_matrix import ConfusionMatrix  # <- torch

# Setup confusion instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")

# Move back to CPU
y_pred_tensor = torch.cat(y_preds).to("cpu")

confmat_tensor = confmat(
    preds=y_pred_tensor, 
    target=test_data.targets  # Labels
)
```

### Options

* binary
* multiclass
* multilabel

## sklearn confusion matrix axes

The return value from `confusion_matrix` &ndash; Is the x axis "Predicted Labels" and the y axis "True Labels"?

Yes, when you use the `confusion_matrix` function from `sklearn.metrics`, the returned confusion matrix is arranged such that the rows represent the true labels and the columns represent the predicted labels.

* x-axis: predicted labels
* y-axis: true labels


```python
from sklearn.metrics import confusion_matrix
import numpy as np

# Example true and predicted labels
y_true = np.array([1, 0, 0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])

# Compute the confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

print(conf_mat)
```

This would output:

```py
array([[2, 2],
       [2, 2]])
```

The first row represents the true labels for the negative class (0) and<br>the second row represents the true labels for the positive class (1).

The first column represents the predicted labels for the negative class and<br>the second column represents the predicted labels for the positive class.

## False positive

So, for example, **`conf_mat[0, 1]`** represents the number of samples that were actually negative **(true label 0)** but were predicted to be positive **(predicted label 1)**, which is 2 in this case.

```py
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

true_negative = cm[0][0]
true_positive = cm[1][1]

false_negative = cm[1][0]
false_positive = cm[0][1]
```

<br>

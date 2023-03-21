## Classification Report

In Python, you can use **scikit-learn's** `classification_report` function to generate a comprehensive classification report for your model's predictions.

To use this function, you'll need the true labels (i.e., `y_test`) and the `predicted` labels.

### Code

Here's an example code snippet that shows how to generate a classification report using `scikit-learn`:

```py
from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_labels))
```

<br>
This will output a report containing evaluation metrics for each class (i.e., digit) in the MNIST dataset.

The report includes precision, recall, F1-score, and support for each class.

The **support** is simply the number of observations in each class.

The **precision** measures how often the model correctly predicted the positive class (i.e., the digit), while the **recall** measures how well the model identified all positive instances.

The **F1-score** is the harmonic mean of precision and recall, and it provides a balanced measure of the model's performance.

### Report output

```py
print(classification_report(y_test, y_pred))
```

```
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.97      0.97      1032
           3       0.98      0.98      0.98      1010
           4       0.97      0.98      0.98       982
           5       0.98      0.98      0.98       892
           6       0.97      0.99      0.98       958
           7       0.97      0.98      0.98      1028
           8       0.97      0.97      0.97       974
           9       0.98      0.96      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
```

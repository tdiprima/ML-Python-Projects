## Classification Report

In Python, you can use **scikit-learn's** `classification_report` function to generate a comprehensive classification report for your model's predictions.

To use this function, you'll need the true labels (i.e., `y_test`) and the `predicted` labels.

Here's an example code snippet that shows how to **generate a classification report using scikit-learn:**

```py
from sklearn.metrics import classification_report

# assuming y_test is the true labels and predicted_labels is the predicted labels
print(classification_report(y_test, predicted_labels))
```

<br>
This will output a report containing evaluation metrics for each class (i.e., digit) in the MNIST dataset.

The report includes precision, recall, F1-score, and support for each class.

The **support** is simply the number of observations in each class.

The **precision** measures how often the model correctly predicted the positive class (i.e., the digit), while the **recall** measures how well the model identified all positive instances.

The **F1-score** is the harmonic mean of precision and recall, and it provides a balanced measure of the model's performance.

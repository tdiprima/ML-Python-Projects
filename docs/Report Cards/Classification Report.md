## Classification Report

You can use scikit-learn's `classification_report` function to generate a comprehensive classification report for your model's predictions.

To use this function, you'll need:

*  True labels (i.e., **`y_test`**)
*  Predicted labels (i.e., **`y_pred`**)

<span style="color:maroon;font-size:larger;font-weight:bold;">Y = Labels</span>

```py
from sklearn.metrics import classification_report

# Generate a classification report
print(classification_report(y_test, predicted_labels))
```

<br>

This will **output a report** containing **evaluation metrics**<br>
<span style="color:#0000dd;font-weight:bold;font-size:larger;">for each class</span> (i.e., <span style="color:#0000dd;font-weight:bold;font-size:larger;">digit</span>) in the MNIST dataset.

**The report includes:**

* Precision
* Recall
* F1-score
* Support for each class

<span style="color:green;font-weight:bold;font-size:larger;">Precision:</span> measures **how often** the model correctly predicted the positive class (i.e., the digit).

<span style="color:purple;font-weight:bold;font-size:larger;">Recall:</span> measures **how well** the model identified all positive instances.

<span style="background:yellow;font-weight:bold;font-size:larger;">F1-score:</span> the **harmonic mean** of precision and recall.

It provides a **balanced measure** of the model's performance.

<span style="color:maroon;font-weight:bold;font-size:larger;">Support:</span> the **number of observations** in each class.

### classification_report

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

### confusion_matrix

```py
print(confusion_matrix(y_test, y_pred))
```

<br>
Output:

```
Confusion matrix:
 [[ 970    0    2    0    0    2    3    1    2    0]
 [   0 1126    5    0    0    0    2    0    2    0]
 [   1    0 1021    1    1    0    2    4    2    0]
 [   0    0    9  987    0    7    0    4    3    0]
 [   0    0   10    0  952    1    8    2    1    8]
 [   3    1    1    8    2  868    5    2    2    0]
 [   3    3    4    0    1    5  941    1    0    0]
 [   0    3   11    0    0    0    0 1006    2    6]
 [   7    1    9   13    3   10    2    6  920    3]
 [   4    6    4   14   12    2    1   29    2  935]]
```
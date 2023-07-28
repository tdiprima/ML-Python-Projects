## Confusion Matrix

A confusion matrix is a way to see how well a machine learning model is doing at classifying things correctly.

### PyTorch ConfusionMatrix

```py
from confusion_matrix import ConfusionMatrix  # <- torch

# Setup confusion instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")

# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)

# Use the function we just made
confmat_tensor = confmat(
    preds=y_pred_tensor, 
    target=test_data.targets  # Labels
)
```

### Options

* binary
* multiclass
* multilabel

## Sklearn confusion\_matrix

<span style="color:#0000dd;">The return value from `confusion_matrix` &ndash; Is the x axis "Predicted Labels" and the y axis "True Labels"?</span>

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

<br>
This would output:

```py
array([[2, 2], [2, 2]])
```

<br>
Negative class (0) and positive class (1).

<center><table border="1"><tr><td>&nbsp;</td><td>predicted labels (0)</td><td>predicted labels (1)</td></tr><tr><td>true labels (0)</td><td align="right">2</td><td align="right">2</td></tr><tr><td>true labels (1)</td><td align="right">2</td><td align="right">2</td></tr></table></center>

<span style="color:#a71f36;">The verbiage messed me up.  It's basically "class" vs "predicted".  True class vs predicted class.  Idk &ndash; it's confusing to explain.</span>

## So here's a fun example

Let's say you have a machine that's supposed to sort apples and oranges into their correct boxes. We'll say apples are the "negative class" (0) and oranges are the "positive class" (1). We are testing how well our machine is working.

Now imagine we gave the machine four fruits to sort: 2 apples and 2 oranges. Once the machine is done sorting, we look at how it did.

<center><table border="1"><tr><td>&nbsp;</td><td>Correct</td><td>Incorrect</td></tr><tr><td><span style="font-size:27px;">🍎</span></td><td align="right">2</td><td align="right">2</td></tr><tr><td><span style="font-size:27px;">🍊</span></td><td align="right">2</td><td align="right">2</td></tr></table></center>

- The first row of our matrix represents how the machine sorted the apples (the "negative class" or 0). The first number in the row (2) tells us that the machine correctly put 2 apples into the apple box. The second number in the row (2) tells us that the machine made a mistake and put 2 apples into the orange box.

- The second row of our matrix represents how the machine sorted the oranges (the "positive class" or 1). The first number in the row (2) tells us that the machine correctly put 2 oranges into the orange box. The second number in the row (2) tells us that the machine made a mistake and put 2 oranges into the apple box.

So the machine correctly sorted 2 apples and 2 oranges, but also mistakenly put 2 apples in the orange box and 2 oranges in the apple box.

This is essentially how the confusion matrix works. It's a way to see how well our "machine" (or in real life, our machine learning model) did at sorting things into the right categories.

## In other words

The rows represent the actual "reality" (***what*** you're predicting).<br>
The columns show what our machine predicted.

The first row of our matrix represents the reality of the <span style="color:#a71f36;">apples</span> (the "negative class" or 0).

The first number in the row (2) tells us that for 2 of these apples, our machine <span style="color:green;">correctly predicted</span> them as apples (put them in the apple box).

The second number in the row (2) tells us that our machine <span style="color:red;">made a mistake</span> and predicted 2 of these apples as oranges (put them in the orange box).

Ditto for <span style="color: orange;">oranges</span>.

So in total, our machine correctly sorted 2 apples and 2 oranges, but also mistakenly thought 2 apples were oranges and 2 oranges were apples.

This is why it's called a "confusion" matrix: it tells us where our machine gets confused!


## False positive

So, for example, `conf_mat[0, 1]` represents the number of samples that were actually negative (true label 0) but were predicted to be positive (predicted label 1), which is 2 in this case.

```py
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

true_negative = cm[0][0]
true_positive = cm[1][1]

false_negative = cm[1][0]
false_positive = cm[0][1]
```

<br>

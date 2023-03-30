## Plot a confusion matrix

I calculated a confusion matrix using scikit-learn `confusion_matrix`.

1. Import the required libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
```

2. Compute the confusion matrix using `confusion_matrix()` function from `sklearn.metrics`.

```python
# Example data
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 1, 0, 2, 2, 0, 1, 2]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

3. Define a function to plot the confusion matrix:

```python
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

4. Call the function `plot_confusion_matrix()` and pass the confusion matrix `cm` and the class labels as arguments.

```python
# Define class labels
class_names = ['Class 0', 'Class 1', 'Class 2']

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
```

The output should be two plots: one for the confusion matrix without normalization and one for the normalized confusion matrix. The plots will show the true and predicted labels on the axes and the number of samples in each category as the color of the corresponding square. If you choose to normalize the matrix, the color will indicate the percentage of samples in each category.

<br>

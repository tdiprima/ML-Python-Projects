"""
Loads a pre-trained model to predict digit labels on the MNIST test dataset, then analyzes
prediction results by creating and visualizing a confusion matrix using a heatmap.

TODO: YOU NEED PRE-TRAINED MODEL my_model.h5
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

loaded_model = keras.models.load_model('models/my_model.h5')

# Set of probabilities for each digit
predictions = loaded_model.predict(x_test)
# predictions[0] is 1 array with 10 things in it.

# y_test.shape: (10000,)
# predictions.shape: (10000, 10)
predictions = np.argmax(predictions, axis=1)  # Pick the column with the best prediction.
# predictions.shape: (10000,) - there you go.

# PLOT PREDICTIONS VS LABELS

# todo: Scatter plot makes no sense
#   CREATE A CONFUSION MATRIX (A.K.A "HEATMAP")

# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# wistia, magma_r, PuOr, YlGn, cmap=mpl.colormaps["PiYG"].resampled(7)

# plt.gcf().canvas.manager.set_window_title("FYI SCATTER PLOT DOESN'T MAKE SENSE FOR MULTI-LABEL CLASSIFICATION")
# plt.title("Multi-Label Classification")
# plt.scatter(y_test, predictions, c=predictions, cmap='plasma')
# plt.xlabel("True outputs")
# plt.ylabel("Predicted outputs")
# plt.colorbar()
# plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
print(f"\ncm type: {type(cm)}")
print(f"\nConfusion matrix:\n{cm}")

import seaborn as sns

sns.set_theme()

fig, ax = plt.subplots()

"""
The returned confusion matrix is arranged such that the rows represent
the true labels and the columns represent the predicted labels.
x-axis = predicted labels
y-axis = true labels
"""
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix")

# cmap='Blues'
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.show()

# import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Load the saved model from a file
loaded_model = keras.models.load_model('models/my_model.h5')

# Use the loaded model to make predictions
predicted_labels = np.argmax(loaded_model.predict(x_test), axis=-1)
# predicted_labels = loaded_model.predict(x_test)  # Hecc no.
# predicted_labels = predictions.argmax(axis=1)  # Almost, but no treato.

"""
Do you really expect me to look through 10K values?
Print to text file.
"""
# with open('y_test.txt', 'w') as filehandle:
#     json.dump(y_test.tolist(), filehandle)
# with open('predictions.txt', 'w') as filehandle:
#     json.dump(predicted_labels.tolist(), filehandle)

"""
Alright, what the hecc is going on here?
"""
# print("test length: {}, type: {}, shape: {}".format(len(y_test), type(y_test), np.shape(y_test)))
# print("pred length: {}, type: {}, shape: {}".format(len(predicted_labels), type(predicted_labels), np.shape(predicted_labels)))

# Messy Output
# for i in range(len(predicted_labels)):
#     # print("Input: {}, True output: {}, Predicted output: {}".format(x_test[i], y_test[i], predicted_labels[i]))
#     print("Predicted output: {}".format(predicted_labels[i]))

# GRAPH
# Plot the true outputs vs. the predicted outputs
plt.scatter(y_test, predicted_labels)
plt.xlabel("True outputs")
plt.ylabel("Predicted outputs")
plt.show()

# y_test is the true labels
# predicted_labels is the predicted labels
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predicted_labels))

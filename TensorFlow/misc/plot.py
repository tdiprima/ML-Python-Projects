import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

loaded_model = keras.models.load_model('models/my_model.h5')

predictions = loaded_model.predict(x_test)

# print("y_test", y_test.shape)  # (10000,)
# print("predictions", predictions.shape)  # (10000, 10)
predictions = np.argmax(predictions, axis=1)  # Our friend, argmax.

# PLOT
plt.gcf().canvas.manager.set_window_title("Multi-class Classification")
plt.title("Multi-label Classification")
# plt.scatter(y_test, predictions)
# plt.scatter(y_test, predictions, c=predictions, cmap='inferno')
plt.scatter(y_test, predictions, c=predictions, cmap='plasma')
plt.xlabel("True outputs")
plt.ylabel("Predicted outputs")
plt.colorbar()
plt.show()

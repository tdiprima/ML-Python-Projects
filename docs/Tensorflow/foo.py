```py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
loaded_model = keras.models.load_model('my_model.h5')
predictions = loaded_model.predict(x_test)
plt.scatter(y_test, predictions)
plt.xlabel("True outputs")
plt.ylabel("Predicted outputs")
plt.show()
```
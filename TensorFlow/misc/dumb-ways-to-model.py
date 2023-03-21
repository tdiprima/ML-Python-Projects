# TODO: Save or load, take your pick.
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define and train a model
model = keras.Sequential([
    # keras.layers.Dense(10, input_shape=(2,)),  # found shape=(32, 28, 28)
    keras.layers.Dense(10, input_shape=(28, 28)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=10)

# Save the model to a file
model.save('my_model.h5')

# Load the saved model from a file
loaded_model = keras.models.load_model('my_model.h5')

# Make predictions with your model
predictions = loaded_model.predict(x_test)

# Print out some results
print('Predictions:', predictions[:5])

# Trains a neural network using the MNIST dataset from TensorFlow and uses the trained model to make predictions on test data.
import tensorflow as tf

# Load data into TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

# Save the model to a file
# model.save('my_model.h5')

# Load the saved model from a file
# loaded_model = keras.models.load_model('my_model.h5')
# Use the loaded model to make predictions
# predictions = loaded_model.predict(x_test)

# Use the trained model to make predictions
predictions = model.predict(x_test)

# Display the results
# for i in range(len(predictions)):
#     print("Input: {}, True output: {}, Predicted output: {}".format(x_test[i], y_test[i], predictions[i]))

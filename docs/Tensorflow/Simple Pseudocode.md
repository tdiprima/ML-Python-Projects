## Tensorflow Pseudocode

TensorFlow is a popular software library for building and training machine learning models. Here's some pseudocode to help explain how TensorFlow works:

1. <span style="color:blue;font-weight:bold">Load</span> data into TensorFlow.
2. <span style="color:green;font-weight:bold">Define</span> the model architecture (what kind of model we want to build).
3. <span style="color:purple;font-weight:bold">Train</span> the model using the loaded data.
4. <span style="color:red;font-weight:bold">Use</span> the trained model to make predictions on new data.

### Load data into TensorFlow 🚚

In order to build a machine learning model, you need data. The first step is to load that data into TensorFlow, so that the model can use it to learn and make predictions.

### Define the model architecture 📝

Once the data is loaded, you need to define the structure of the machine learning model that you want to build. This includes things like:

* the number and type of layers
* the activation functions to use
* the optimization algorithm to use

### Train the model 🐕

With the data and the model architecture defined, it's time to start training the model.

This involves:

* feeding the data into the model, 
* letting the model make predictions, and
* then adjusting the model's parameters (weights and biases) based on how well those predictions match the actual data.

This process is repeated multiple times, with the goal of **minimizing the difference** between the **predicted values** and the **actual values** in the data set.

### Use the trained model to make predictions 🎾

Once the model is trained, you can use it to make predictions on new data.

This involves:

* feeding the new data into the model, and
* letting it make predictions based on what it learned during training.

### Build a neural network that can classify handwritten digits from the MNIST data set

```py
# Load data into TensorFlow
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define the model architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

# Train the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)

# Use the trained model to make predictions
predictions = model.predict(x_test)
```

<br>

* We start by loading the data into TensorFlow, then 
* defining the structure of the neural network:
    * (two layers, with the second layer having 10 output neurons, one for each possible digit).

* We then train the model using the training data, and finally
* make predictions on the test data.

<br>

## Sequential Model

```py
# Define the model architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
```

### do we use a Sequential model most of the time?  how do we know to use that one?

The `Sequential` model is one of the most commonly used models in TensorFlow, especially for beginners. The `Sequential` model allows you to create a neural network by simply stacking layers on top of each other, making it easy to define the structure of your model.

You can use the `Sequential` model when your neural network has a linear topology, meaning that the output of one layer is connected to the input of the next layer in a sequential order. This is the most common type of neural network, where data flows through the layers in a simple sequence.

However, if your neural network has a more complex structure, such as multiple inputs or outputs, or if the layers have more complex connections between them, then you may need to use other types of models, such as the functional API or the subclassing API.

So, to summarize, the `Sequential` model is a good choice when you have a simple neural network with a linear topology. However, as your neural network becomes more complex, you may need to use other types of models to achieve your goals.

## "Flatten" and "Dense"

In machine learning, we often use neural networks to learn patterns in data. A neural network is made up of multiple layers of interconnected neurons that transform the input data in a way that allows the network to learn useful representations of the data.

The `Flatten` layer is used to convert multi-dimensional input data, such as images or time series data, into a one-dimensional array. This is necessary because most neural network layers can only work with one-dimensional data.

Let me give you an example. Suppose we have an image of size 28x28 pixels, which we want to feed into a neural network. This image can be represented as a 2D array of 28 rows and 28 columns. However, most neural network layers expect one-dimensional data, so we need to flatten this 2D array into a one-dimensional array of length 784 (28 x 28). The `Flatten` layer does exactly that - it takes the multi-dimensional input data and flattens it into a one-dimensional array.

The `Dense` layer is used to connect each neuron in one layer to every neuron in the next layer. The `Dense` layer is called dense because every neuron in one layer is connected to every neuron in the next layer.

The `Dense` layer is where the neural network learns to identify patterns in the input data. Each neuron in a Dense layer receives input from every neuron in the previous layer and computes a weighted sum of those inputs. The output of each neuron is then passed through an activation function, which determines whether the neuron should fire (output a 1) or not (output a 0).

Let's take the example of a neural network that is used to classify images of handwritten digits. The Dense layer in this network takes the flattened image as input and learns to identify patterns in the image that correspond to a particular digit. For example, a Dense layer might learn to identify the curved lines and loops that are characteristic of the digit "2". The output of the Dense layer is then passed to another Dense layer or a final output layer that produces a prediction for the digit in the image.

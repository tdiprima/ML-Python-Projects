import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(tf.__version__)

# Generate some random data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Define the model variables
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# Define the model
def model(x):
    return W * x + b

# Define the loss function (mean squared error)
def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Define the optimizer (gradient descent)
optimizer = tf.optimizers.SGD(learning_rate=0.5)

# Training loop
for step in range(201):
    with tf.GradientTape() as tape:
        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if step % 20 == 0:
        print(step, W.numpy(), b.numpy())

# Plot the results
plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, model(x_data), label='Fitted line')
plt.legend()
plt.show()

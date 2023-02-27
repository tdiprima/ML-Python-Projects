"""
Instructions for updating:
Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.
from tensorflow.examples.tutorials.mnist import input_data
"""
import tensorflow as tf


# NEW input function
def input_fn():
    # TODO: Note
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    assert train_images.shape == (60000, 28, 28)
    assert train_labels.shape == (10000, 28, 28)
    assert test_images.shape == (60000,)
    assert test_labels.shape == (10000,)

    # Normalize the pixel values
    train_images = train_images / 255.0
    test_images = test_images / 255.0  # TODO: test_images not used

    # Create a dataset from the training set
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(100)
    dataset = dataset.prefetch(1)  # To resolve make_one_shot_iterator

    return dataset


# ORIGINAL input function
def input_fn1():
    # Load the MNIST dataset
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # TODO: we lost the "one hot" part
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Normalize the pixel values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Create a dataset from the training set
    # dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels))

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(100)
    dataset = dataset.repeat()

    # Create an iterator for the dataset
    # WITH Original: 'RepeatDataset' object has no attribute 'make_one_shot_iterator'
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of inputs and labels
    inputs, labels = iterator.get_next()

    return inputs, labels


# Define the neural network model
def model_fn(features, labels, mode):
    # Define the layers of the model
    input_layer = tf.reshape(features, [-1, 784])
    hidden_layer = tf.layers.dense(inputs=input_layer, units=256, activation=tf.nn.relu)
    output_layer = tf.layers.dense(inputs=hidden_layer, units=10)

    # Define the predictions and loss
    predictions = tf.argmax(output_layer, axis=1)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=output_layer)

    # Define the training operation
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # Define the evaluation metrics
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions)
    }

    # Return the EstimatorSpec
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


try:
    # Create the estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn)
except Exception as e:
    print("\nCreate estimator:", e)
    exit(1)

try:
    # Train the model
    estimator.train(input_fn=input_fn, steps=10000)
except Exception as ex:
    # WITH New: module 'tensorflow' has no attribute 'layers' 
    print("\nTrain model:", ex)
    exit(1)

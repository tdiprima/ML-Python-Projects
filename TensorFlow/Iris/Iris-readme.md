## Load Iris with TensorFlow

See: <a href="../../docs/Backpropagation.md">Backpropagation.md</a> and <a href="un_iris_es_una_flor.py">un iris es una flor</a>

<span style="color:#0000dd;">I'm trying to load the Iris dataset using tensorflow like this: iris = tf.keras.datasets.iris; but I get an error saying "module 'keras.api._v2.keras.datasets' has no attribute 'iris'".</span>

It seems like you are trying to load the Iris dataset using TensorFlow's Keras API, but the specific dataset attribute you are using (`iris`) is not available in TensorFlow's Keras datasets module.

In TensorFlow, the Iris dataset is available through the `sklearn.datasets` module rather than directly in TensorFlow's Keras API. To load the Iris dataset, you should use scikit-learn (sklearn).

Here's how you can load the Iris dataset using scikit-learn:

<a href="https://github.com/tdiprima/kraddy/blob/master/TensorFlow/Iris/un_iris_es_una_flor.py#L6">sklearn.datasets load_iris</a>

<br>
Make sure you have scikit-learn installed in your Python environment before running the above code. You can install it using pip:

```sh
pip install scikit-learn
```

<br>
If you specifically need to load the dataset through TensorFlow's Keras API, you can consider using other datasets available in TensorFlow, such as MNIST or CIFAR-10. But if you want to work with the Iris dataset, using scikit-learn as shown above is a common approach.

## Keras Datasets

Looking at: `/usr/local/anaconda3/lib/python3.9/site-packages/keras/api/_v2/keras/datasets/__init__.py`

This what we got:

```py
from keras.api._v2.keras.datasets import
boston_housing
cifar10
cifar100
fashion_mnist
imdb
mnist
reuters
```

Which we would load like this:

```py
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.[thing].load_data()
```



## Float 32

<span style="color:#0000dd;">For the iris dataset, do I have to do: x\_train.astype(np.float32)?</span>

In most cases, you don't need to explicitly convert the features (input data) of the Iris dataset to `np.float32` if you are using the scikit-learn `load_iris` function to load the dataset. The `load_iris` function already returns the features as a NumPy array of floating-point numbers (`np.float64`).

Here's an example of loading the Iris dataset using scikit-learn:

<a href="https://github.com/tdiprima/kraddy/blob/master/TensorFlow/Iris/un_iris_es_una_flor.py#L11">iris.data</a>

Since the features are already in `np.float64` format, there is generally no need to explicitly convert them to `np.float32`. NumPy arrays usually perform well with `np.float64`, and in most cases, you can work directly with them as they are.

However, if you have a specific need or reason to work with `np.float32` (e.g., for memory efficiency in certain cases or compatibility with specific libraries), you can convert the features to `np.float32` using the `astype` method as you mentioned:

```python
X_train = X_train.astype(np.float32)
```

<br>

But again, for the Iris dataset, it is not necessary for standard machine learning tasks with scikit-learn. You can use the features directly as they are loaded from the `load_iris` function.

## To categorical 3

<span style="color:#0000dd;">ok.  it looks like the data has 4 features.  Some other programmer (Henry!) preprocessed the data like: y\_train = tf.keras.utils.to\_categorical(y\_train, num_classes=3)`.  I think this is wrong...</span>

You are correct. The preprocessing step you mentioned, `tf.keras.utils.to_categorical(y_train, num_classes=3)`, is incorrect for the Iris dataset. The Iris dataset is a multiclass classification problem with three classes, not a one-hot encoded problem.

The `to_categorical` function from `tf.keras.utils` is used to convert integer labels into one-hot encoded format, which is useful for tasks where you have multiple classes, and each sample can belong to only one class.

However, in the case of the Iris dataset, the target values (labels) are already encoded as integers representing three different classes (0, 1, and 2). These integer labels directly correspond to the three classes: 'setosa', 'versicolor', and 'virginica'.

Here's the correct way to load and preprocess the Iris dataset for a multiclass classification task:

<a href="https://github.com/tdiprima/kraddy/blob/master/TensorFlow/Iris/un_iris_es_una_flor.py#L20">train\_test\_split</a>

~~Since the target values `y_train` and `y_test` are already integers representing the class labels, you can directly use them for multiclass classification tasks without further preprocessing using `to_categorical`.~~ &lt;&mdash; That's not correct.  And that's why we're getting the error below:

## Value (shapes) error

<span style="color:#0000dd;">Aforementioned programmer coded the training and eval incorrectly (please see code below).  I get a ValueError "Shapes (32,) and (32, 3) are incompatible" on this line: `loss = loss_fn(y_batch, y_pred)`.

The issue in the code is that the target values `y_train` and `y_test` are not one-hot encoded, but the model's output is returning a probability distribution for each sample, which is expected for a multiclass classification problem. The model is outputting a shape of `(batch_size, num_classes)` where `num_classes=3` for the Iris dataset.

To resolve this issue, you need to convert the target values `y_train` and `y_test` to one-hot encoded format using `tf.keras.utils.to_categorical` to match the shape of the model's output.

<a href="https://github.com/tdiprima/kraddy/blob/master/TensorFlow/Iris/un_iris_es_una_flor.py#L18">to\_categorical</a>

With this change, the target values `y_train` and `y_test` will be in one-hot encoded format, and the code should run without any ValueError. The model's output (`y_pred`) will also have the same shape as the one-hot encoded targets, allowing you to calculate the loss correctly.

## Visualize, visualize, visualize

<span style="color:#0000dd;">If I wanted to plot the test\_loss's versus the test\_accuracy's in a graph, how could I do that?  I probably needed to save those values all along?  Or does TensorFlow save that information in the model or something?</span> <span style="color:red;">Like: History!</span><span style="color:#0000dd;">(TODO: Training loss, validation loss.)</span>

To plot the `test_loss` versus the `test_accuracy` over the epochs, you can create lists to store these values during training and then use them for visualization. TensorFlow does not automatically save these values during training, so you need to explicitly collect and store them.

Here's how you can modify your code to save the `test_loss` and `test_accuracy` values during training:

<a href="https://github.com/tdiprima/kraddy/blob/master/TensorFlow/Iris/un_iris_es_una_flor.py#L58">test\_loss, test\_accuracy, test\_accuracies</a>

After training, you can use `test_losses` and `test_accuracies` to create a plot using libraries like Matplotlib. Here's a simple example of how to plot these values:

<a href="https://github.com/tdiprima/kraddy/blob/master/TensorFlow/Iris/un_iris_es_una_flor.py#L66">plt.plot</a>

This will create a plot showing the trend of `test_loss` and `test_accuracy` values over the epochs, which can help you analyze the model's performance during training.

## Model, loss fn, optimizer

<span style="color:#0000dd;">What model and loss function should I use for the Iris data?</span>

For the Iris dataset, a common approach is to use a simple classification model such as a fully connected neural network. Since the Iris dataset is a multiclass classification problem with three classes ('setosa', 'versicolor', and 'virginica'), you should use a softmax activation function in the final layer of the model.

Here's an example of how to create a simple fully connected neural network model for the Iris dataset:

<a href="fully-connected.py">fully-connected.py</a>

### Note: keras.callbacks.History

In this example, the model has two hidden layers with 64 and 32 units, respectively. The final layer has 3 units with a softmax activation function, which outputs a probability distribution over the three classes.

The loss function used here is `'categorical_crossentropy'`, which is suitable for multiclass classification problems with one-hot encoded target values. The `'adam'` optimizer is used for training, which is a popular choice for gradient-based optimization.

After training the model, you can evaluate its performance using the test set and plot the test loss and test accuracy as shown in the previous example.

<br>

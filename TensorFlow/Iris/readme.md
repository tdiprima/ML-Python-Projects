## Load Iris with TensorFlow

See: docs/Backpropagation.md

<span style="color:#0000dd;">I'm trying to load the Iris dataset using tensorflow like this: iris = tf.keras.datasets.iris; but I get an error saying "module 'keras.api._v2.keras.datasets' has no attribute 'iris'".</span>

It seems like you are trying to load the Iris dataset using TensorFlow's Keras API, but the specific dataset attribute you are using (`iris`) is not available in TensorFlow's Keras datasets module.

In TensorFlow, the Iris dataset is available through the `sklearn.datasets` module rather than directly in TensorFlow's Keras API. To load the Iris dataset, you should use scikit-learn (sklearn).

Here's how you can load the Iris dataset using scikit-learn:

```python
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Access the features and labels
X, y = iris.data, iris.target
```

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

## Smallest Dataset

<span style="color:#0000dd;">Which is the smallest dataset out of keras's boston\_housing, cifar10, cifar100, fashion\_mnist, imdb, mnist, reuters?</span>

Among the datasets you listed, the smallest dataset is `imdb`, which stands for the Internet Movie Database. It is a dataset used for sentiment analysis, containing movie reviews and their associated sentiment labels (positive or negative).

The size of the `imdb` dataset is relatively small compared to the other datasets you mentioned. It contains a total of 50,000 movie reviews, split into 25,000 reviews for training and 25,000 reviews for testing.

Here's how you can load the `imdb` dataset using TensorFlow's Keras API:

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```

<br>

The `num_words=10000` argument limits the dataset to the top 10,000 most frequently occurring words in the reviews. This is done to keep the dataset manageable while preserving most of the relevant information for sentiment analysis.

## IMDb Features

<span style="color:red;">question - so what are these 4 features in the imdb dataset?</span>

*I was wrong &ndash; 4 was the Iris dataset.*

In the IMDb dataset, each data point (movie review) consists of a sequence of words from the review's text. Each word is represented by an integer index, which corresponds to the word's position in a word index dictionary.

The IMDb dataset does not have traditional features like numerical or categorical columns, as in other datasets. Instead, the features are sequences of integers, where each integer represents a specific word in the movie review.

To summarize, the four features in the IMDb dataset are:

1. Integer sequences representing the words in the movie review.
2. Each integer represents a specific word from the word index dictionary.
3. The sequence length may vary from one review to another.
4. The `num_words` argument (e.g., `num_words=10000`) during loading limits the dataset to the top N most frequently occurring words, where N is the value of `num_words`. This is done to manage the dataset size and remove less common words.

The target variable (label) is a binary sentiment value: 0 for negative sentiment and 1 for positive sentiment. The goal is to predict the sentiment (positive or negative) based on the sequence of words in the review.

## Train test split

<span style="color:#0000dd;">In what order does sklearn.model\_selection's train\_test\_split return values?</span>

`sklearn.model_selection`'s `train_test_split` function returns the values in the following order:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<br>

1. **`X_train`**: This variable contains the training data features (input variables) that will be used to train your machine learning model. It is a subset of the original `X` data.

2. **`X_test`**: This variable contains the testing data features (input variables) that will be used to evaluate the performance of your trained model. It is also a subset of the original `X` data.

3. **`y_train`**: This variable contains the corresponding target values (labels) for the training data. It is a subset of the original `y` data.

4. **`y_test`**: This variable contains the corresponding target values (labels) for the testing data. It is also a subset of the original `y` data.

<br>

The `train_test_split` function is commonly used to split a dataset into training and testing sets. The `test_size` parameter determines the proportion of the data that will be used for testing (e.g., `test_size=0.2` means 20% of the data will be used for testing). The `random_state` parameter is optional and is used to set a random seed for reproducibility.

## Float 32

<span style="color:#0000dd;">For the iris dataset, do I have to do: x\_train.astype(np.float32)?</span>

In most cases, you don't need to explicitly convert the features (input data) of the Iris dataset to `np.float32` if you are using the scikit-learn `load_iris` function to load the dataset. The `load_iris` function already returns the features as a NumPy array of floating-point numbers (`np.float64`).

Here's an example of loading the Iris dataset using scikit-learn:

```python
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# The features (input data) are already in np.float64 format
X = iris.data

# The target values (labels) are in np.int64 format
y = iris.target
```

<br>

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

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# No need for one-hot encoding; y_train and y_test are already integer labels
# Proceed with your machine learning model training and evaluation
```

<br>

Since the target values `y_train` and `y_test` are already integers representing the class labels, you can directly use them for multiclass classification tasks without further preprocessing using `to_categorical`.

## Value (shapes) error

<span style="color:#0000dd;">Aforementioned programmer coded the training and eval incorrectly (please see code below).  I get a ValueError "Shapes (32,) and (32, 3) are incompatible" on this line: `loss = loss_fn(y_batch, y_pred)`.  Here is the code:</span> 

```py
# Train the model
batch_size = 32
epochs = 100
for epoch in range(epochs):
    # Floor division
    for batch in range(len(x_train) // batch_size):
        start = batch * batch_size
        end = start + batch_size
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]

        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Evaluate the model on the test set
    y_pred = model(x_test)
    test_loss = loss_fn(y_test, y_pred)
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()(y_test, y_pred)
    print(f'Epoch {epoch + 1}, loss={loss:.2f}, test_loss={test_loss:.2f}, test_accuracy={test_accuracy:.2f}')

```

<br>

The issue in the code is that the target values `y_train` and `y_test` are not one-hot encoded, but the model's output is returning a probability distribution for each sample, which is expected for a multiclass classification problem. The model is outputting a shape of `(batch_size, num_classes)` where `num_classes=3` for the Iris dataset.

To resolve this issue, you need to convert the target values `y_train` and `y_test` to one-hot encoded format using `tf.keras.utils.to_categorical` to match the shape of the model's output. Here's how you can fix the code:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert target values to one-hot encoded format
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define your model and loss function (assuming you have already done this)

# Train the model
batch_size = 32
epochs = 100
for epoch in range(epochs):
    # Floor division
    for batch in range(len(X_train) // batch_size):
        start = batch * batch_size
        end = start + batch_size
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Evaluate the model on the test set
    y_pred = model(X_test)
    test_loss = loss_fn(y_test, y_pred)
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()(y_test, y_pred)
    print(f'Epoch {epoch + 1}, loss={loss:.2f}, test_loss={test_loss:.2f}, test_accuracy={test_accuracy:.2f}')
```

<br>

With this change, the target values `y_train` and `y_test` will be in one-hot encoded format, and the code should run without any ValueError. The model's output (`y_pred`) will also have the same shape as the one-hot encoded targets, allowing you to calculate the loss correctly.

## Visualize, visualize, visualize

<span style="color:#0000dd;">If I wanted to plot the test\_loss's versus the test\_accuracy's in a graph, how could I do that?  I probably needed to save those values all along?  Or does TensorFlow save that information in the model or something?</span> <span style="color:red;">Like: History!</span><span style="color:#0000dd;">(TODO: Training loss, validation loss.)</span>

To plot the `test_loss` versus the `test_accuracy` over the epochs, you can create lists to store these values during training and then use them for visualization. TensorFlow does not automatically save these values during training, so you need to explicitly collect and store them.

Here's how you can modify your code to save the `test_loss` and `test_accuracy` values during training:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert target values to one-hot encoded format
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define your model and loss function (assuming you have already done this)

# Train the model
batch_size = 32
epochs = 100

# Lists to store test_loss and test_accuracy values
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    # Floor division
    for batch in range(len(X_train) // batch_size):
        start = batch * batch_size
        end = start + batch_size
        x_batch = X_train[start:end]
        y_batch = y_train[start:end]

        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Evaluate the model on the test set
    y_pred = model(X_test)
    test_loss = loss_fn(y_test, y_pred)
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()(y_test, y_pred)
    print(f'Epoch {epoch + 1}, loss={loss:.2f}, test_loss={test_loss:.2f}, test_accuracy={test_accuracy:.2f}')

    # Save test_loss and test_accuracy for plotting
    test_losses.append(test_loss.numpy())
    test_accuracies.append(test_accuracy.numpy())
```

<br>

After training, you can use `test_losses` and `test_accuracies` to create a plot using libraries like Matplotlib. Here's a simple example of how to plot these values:

```python
import matplotlib.pyplot as plt

# Plot test_loss versus test_accuracy
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title('Test Loss and Test Accuracy over Epochs')
plt.grid(True)
plt.show()
```

<br>

This will create a plot showing the trend of `test_loss` and `test_accuracy` values over the epochs, which can help you analyze the model's performance during training.

## Model, loss fn, optimizer

<span style="color:#0000dd;">What model and loss function should I use for the Iris data?</span>

For the Iris dataset, a common approach is to use a simple classification model such as a fully connected neural network. Since the Iris dataset is a multiclass classification problem with three classes ('setosa', 'versicolor', and 'virginica'), you should use a softmax activation function in the final layer of the model.

Here's an example of how to create a simple fully connected neural network model for the Iris dataset:

```python
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert target values to one-hot encoded format
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model with an appropriate loss function for multiclass classification
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 100
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
```

### Note: keras.callbacks.History

In this example, the model has two hidden layers with 64 and 32 units, respectively. The final layer has 3 units with a softmax activation function, which outputs a probability distribution over the three classes.

The loss function used here is `'categorical_crossentropy'`, which is suitable for multiclass classification problems with one-hot encoded target values. The `'adam'` optimizer is used for training, which is a popular choice for gradient-based optimization.

After training the model, you can evaluate its performance using the test set and plot the test loss and test accuracy as shown in the previous example.

<br>

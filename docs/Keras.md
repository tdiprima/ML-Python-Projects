## Keras Image Classification

```py
# Import required libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Define model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Evaluate model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

<br>
This program uses the **Keras library** to 

build a convolutional neural network (**CNN**) model 

to classify images of **handwritten digits** 

from the **MNIST** dataset.

* Imports the required libraries
* Loads the data
* Preprocesses it
* Defines the model architecture
* Compiles the model
* Trains the model on the training data
* Evaluates the model on the testing data

(OK, so that's step 1.  Then we wanna try it out on real world data.)

### Pro-tip!

When you get:

```
"ModuleNotFoundError: No module named 'keras.models'; 'keras' is not a package"
```

<br>
<span style="color:red;font-size:larger">"Dr. Snoopy" says:</span>

The problem is that you have a file named **"keras.py"** and this shadows the real keras package.

Don't do that, never name a python script the same as a package.

Solution is to **rename your script.**

I knew that. 🤦‍♀️

## Output

10 epochs.

Loss. Accuracy. val\_loss. val\_accuracy.

Final Accuracy.

```c
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

Epoch 1/10
300/300 [==============================] - 8s 25ms/step - loss: 0.2728 - accuracy: 0.9240 - val_loss: 0.1027 - val_accuracy: 0.9719

Etc.

Epoch 10/10
300/300 [==============================] - 10s 34ms/step - loss: 0.0098 - accuracy: 0.9974 - val_loss: 0.0392 - val_accuracy: 0.9880

Accuracy: 98.80%
```

<br>

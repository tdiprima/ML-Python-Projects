## Keras Image Classification

### Sure, here is an example of pseudocode for a typical Keras program for image classification:

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

This program uses the Keras library to build a convolutional neural network (CNN) model to classify images of handwritten digits from the MNIST dataset.

The program imports the required libraries, loads the data, preprocesses it, defines the model architecture, compiles the model, trains the model on the training data, and evaluates the model on the testing data.

(OK, so that's step 1.  Then we wanna try it out on real world data.)

### Pro-tip!

When you get:

```
"ModuleNotFoundError: No module named 'keras.models'; 'keras' is not a package"
```

"Dr. Snoopy" says:

The problem is that you have a file named "keras.py" and this shadows the real keras package. Don't do that, never name a python script the same as a package.

Solution is to rename your keras.py script into a different name.

I knew that. 🤦‍♀️

## Output

10 epochs.

Loss. Accuracy. val\_loss. val\_accuracy.

Final Accuracy.

```haskell
2023-02-20 10:11:59.659725: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 1s 0us/step
2023-02-20 10:12:06.070775: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/10
300/300 [==============================] - 8s 25ms/step - loss: 0.2728 - accuracy: 0.9240 - val_loss: 0.1027 - val_accuracy: 0.9719
Epoch 2/10
300/300 [==============================] - 8s 28ms/step - loss: 0.0829 - accuracy: 0.9762 - val_loss: 0.0624 - val_accuracy: 0.9796
Epoch 3/10
300/300 [==============================] - 9s 29ms/step - loss: 0.0549 - accuracy: 0.9840 - val_loss: 0.0545 - val_accuracy: 0.9821
Epoch 4/10
300/300 [==============================] - 9s 29ms/step - loss: 0.0417 - accuracy: 0.9875 - val_loss: 0.0425 - val_accuracy: 0.9856
Epoch 5/10
300/300 [==============================] - 10s 32ms/step - loss: 0.0322 - accuracy: 0.9906 - val_loss: 0.0445 - val_accuracy: 0.9849
Epoch 6/10
300/300 [==============================] - 10s 33ms/step - loss: 0.0255 - accuracy: 0.9925 - val_loss: 0.0423 - val_accuracy: 0.9858
Epoch 7/10
300/300 [==============================] - 11s 37ms/step - loss: 0.0197 - accuracy: 0.9945 - val_loss: 0.0387 - val_accuracy: 0.9870
Epoch 8/10
300/300 [==============================] - 13s 43ms/step - loss: 0.0175 - accuracy: 0.9948 - val_loss: 0.0406 - val_accuracy: 0.9867
Epoch 9/10
300/300 [==============================] - 13s 42ms/step - loss: 0.0130 - accuracy: 0.9963 - val_loss: 0.0395 - val_accuracy: 0.9877
Epoch 10/10
300/300 [==============================] - 10s 34ms/step - loss: 0.0098 - accuracy: 0.9974 - val_loss: 0.0392 - val_accuracy: 0.9880
Accuracy: 98.80%
```
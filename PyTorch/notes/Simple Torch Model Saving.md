## Save model

```python
import torch

# define your model
model = ...

# define the path to save the model
PATH = "my_model.pth"

# save the model state dictionary
torch.save(model.state_dict(), PATH)
```


## Load model

```python
import torch

# define the path to the saved model
PATH = "my_model.pth"

# define your model class
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # define your model layers here

# create an instance of your model class
model = MyModel()

# load the saved model parameters into the instance
model.load_state_dict(torch.load(PATH))
```

<br>

<mark>If your saved model was trained on a **GPU** and you are now loading it on a **CPU**,</mark> you can use the following code:

```python
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))
```

<br>

When you load a pre-trained PyTorch model, you are essentially loading the trained weights of the model into memory. However, you still need to define the architecture of the model so that the weights can be applied to the correct layers and operations.

In other words, loading a pre-trained model is only part of the process.

**(Serialize = save; deserialize = load.)**

## Fully connected layers

```python
import torch
import torch.nn as nn

# Define a simple neural network architecture
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Load a pre-trained model
pretrained_model = torch.load('model.pt')

# Create an instance of the model class
model = MyModel()

# Assign the pre-trained weights to the model
model.load_state_dict(pretrained_model)

# Use the model for prediction or other tasks
inputs = torch.randn(1, 10)
outputs = model(inputs)
```

In deep learning, "fc" typically stands for **"fully connected" layers.**

A fully connected layer is a type of layer where each neuron in the layer is connected to every neuron in the previous layer.

In the `MyModel` class example, `self.fc1` is an instance of the `nn.Linear` class, which represents a fully connected layer in PyTorch.

Specifically, it is a fully connected layer with 10 input neurons (i.e., the number of features in the input data) and 5 output neurons.

The second fully connected layer `self.fc2` has 5 input neurons (i.e., the output of `self.fc1`) and 1 output neuron (i.e., the output of the entire model).

Fully connected layers are commonly used in neural network architectures for tasks such as image **classification**, natural language processing **(NLP),** and other machine learning tasks where the input data has a **fixed number of features.**


## Automatically reconstructed

In Tensorflow (Keras), all you have to do is `load_model('my_model.h5')`, and you're good to go.

In TensorFlow (specifically, in the Keras API), **the model architecture is saved along with the weights** when you save the model to a file.

So, when you load the model using `load_model('my_model.h5')`, the model architecture is **automatically reconstructed** along with the **saved weights**.

```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define a simple neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Train the model and save it to a file
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=10)
model.save('my_model.h5')

# Load the saved model and use it for prediction
loaded_model = load_model('my_model.h5')
loaded_model.predict(X_test)
```

<br>

<!--The instructions for how to save a model using torch.save seem unreasonably difficult.  Can you write me a simple code example?

Certainly, here's an example of how to save a PyTorch model using torch.save():-->

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

In this example, `model` represents your PyTorch model.


## Load model
<!--Great.  Now, how do I read in a saved model?

To load a saved PyTorch model using `torch.load()`, you can use the following code:-->

```python
import torch

# define the path to the saved model
PATH = "my_model.pth"

# create an instance of your model class
model = ...

# load the saved model parameters into the instance
model.load_state_dict(torch.load(PATH))
```

In this code, `PATH` should be the path to the file where your model was saved.

The `model` variable should be an instance of the same **model class** that you used to create the saved model.

By calling `model.load_state_dict(torch.load(PATH))`, you're loading the **saved state dictionary** into your model instance.

### GPU

Note that if your saved model was trained on a GPU, you may need to pass the **`map_location`** argument to `torch.load()` to ensure that the saved model parameters are loaded correctly.

For example, if your saved model was trained on a **GPU** and you're loading it on a **CPU**, you can use the following code:

```python
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH, map_location=device))
```

This will ensure that the saved model parameters are loaded onto the CPU, rather than trying to load them onto a non-existent GPU.

## Meh... I beg your pa&ndash; 🐰

<!-- OK.  Before we do model.load_state_dict(torch.load(PATH)), we create an instance of our model class.  How do I create an instance?  I imagine it can be an "empty" instance? -->

Okay, how do I create a plain old instance?

Yes, you can create an instance of your model class **without any trained parameters** by calling the constructor of your model class.

```python
import torch.nn as nn

# define your model class
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # define your model layers here

# create an instance of your model class
model = MyModel()
```

In this example, `MyModel` is a subclass of `nn.Module` that defines the layers of your model.

By calling `model = MyModel()`, you're creating an instance of your model with the default values of its parameters.

You can then load the saved parameters into this instance using `model.load_state_dict(torch.load(PATH))`, as described in the previous answer.

### TODO

Check this out, because maybe I can save the entire model.  (Serialize = save; deserialize = load.)

[save-load-entire-model](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model)

## Double work?

Why do you still need to create an instance of your model class, even when loading a pre-trained PyTorch model?

When you load a pre-trained PyTorch model, you are essentially loading the trained weights of the model into memory. However, you still need to define the architecture of the model so that the weights can be applied to the correct layers and operations.

In other words, loading a pre-trained model is only part of the process.

You still need to create an instance of the model class and define its architecture before you can use the pre-trained weights to make predictions or perform other tasks.

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

In this example, we define a simple neural network architecture in the `MyModel` class.

We then load a pre-trained model from a file using `torch.load()`.

However, we still need to create an instance of `MyModel` and assign the pre-trained weights to it, using `model.load_state_dict()`, before we can use it for prediction or other tasks.

## fc? 🧐

When we define the `MyModel` class, we use a variable called `self.fc1`.  What does "fc" stand for?

In deep learning, "fc" typically stands for **"fully connected" layers.**

A fully connected layer is a type of layer where each neuron in the layer is connected to every neuron in the previous layer.

In the `MyModel` class example, `self.fc1` is an instance of the `nn.Linear` class, which represents a fully connected layer in PyTorch.

Specifically, it is a fully connected layer with 10 input neurons (i.e., the number of features in the input data) and 5 output neurons.

The second fully connected layer `self.fc2` has 5 input neurons (i.e., the output of `self.fc1`) and 1 output neuron (i.e., the output of the entire model).

Fully connected layers are commonly used in neural network architectures for tasks such as image classification, natural language processing, and other machine learning tasks where the input data has a fixed number of features.

<hr>

## No double work.

PyTorch 🆚 Tensorflow / Keras

That makes sense.  In contrast, why is it that in Tensorflow (Keras), all you have to do is `load_model('my_model.h5')`, and you're good to go?

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

In this example, we define a simple neural network architecture using the Keras API.

We then train the model and save it to a file using `model.save('my_model.h5')`.

When we load the saved model using `load_model('my_model.h5')`, the model architecture is automatically reconstructed along with the saved weights, so we can use the loaded model for prediction without having to define the architecture again.

So, to summarize, in **TensorFlow (Keras)**, the model architecture is saved along with the weights, 

whereas in **PyTorch**, the weights are saved separately from the model architecture.

This is why the process of loading a pre-trained model is different between the two frameworks.

### But why?

I'm gonna say they designed it to be flexible, so that's why they separated it.

<br>

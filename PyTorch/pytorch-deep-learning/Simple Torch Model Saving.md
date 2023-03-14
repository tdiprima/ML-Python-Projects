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

Okay, how do I create an plain old instance?

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

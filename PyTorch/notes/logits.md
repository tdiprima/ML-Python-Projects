## Cross entropy with logits

Cross entropy with logits is a common loss function used in classification tasks. In PyTorch, you can calculate it using `nn.CrossEntropyLoss()`:

```python
import torch
import torch.nn as nn

# Define your model output and ground truth
logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]])
labels = torch.tensor([0, 1, 2])

# Create a CrossEntropyLoss object
loss_fn = nn.CrossEntropyLoss()

# Calculate the loss
loss = loss_fn(logits, labels)

print(loss.item())  # 2.4076058864593506
```

In this example, `logits` represents the output of your model, and `labels` represent the ground truth labels.

The `CrossEntropyLoss` object takes care of applying the **softmax** function to the logits and **calculating** the cross-entropy loss.

The `loss.item()` call is used to get the actual loss value as a scalar tensor.

## Logits in TensorFlow

[What is the meaning of the word logits in TensorFlow?](https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow)

In the following TensorFlow function, we must feed the activation of artificial neurons in the final layer.  *"Activation" meaning "softmax", I guess.*

```ruby
# TensorFlow
loss_function = tf.nn.softmax_cross_entropy_with_logits(
     logits = last_layer,
     labels = target_output
)
```

### Hecc. What the fluff does this mean?

<span style="color:#997fff;">"In context of deep learning, the logits layer means the layer that feeds in to softmax (or other such normalization). The output of the softmax are the probabilities for the classification task, and its input is logits layer."</span>

Okay, so let's imagine you're trying to teach a computer to recognize pictures of cats and dogs.

When the computer looks at a picture of a cat or a dog, it needs to decide which one it is. It does this by looking at different parts of the picture and trying to figure out what they mean.

But how does the computer know if it's right or wrong? That's where the "logits layer" comes in. This layer is **like a bridge** between the computer's guess and the answer you want it to give.

**The logits layer** takes all the information the computer has gathered from the picture and turns it into a set of numbers. These numbers aren't probabilities yet, they're just a bunch of random values that represent **how confident** the computer is in its guess.

Next, these numbers get sent to another layer called the **"softmax" layer.** This layer takes those random values and turns them into **probabilities** - basically, it decides how likely it is that the picture is a cat or a dog.

So, the **"logits layer"** is just a step in the process of teaching the computer to recognize pictures. It's like a middleman that helps turn the computer's observations into a final answer.

<hr>

**deep learning:** is a fancy way of saying you're teaching the computer to learn by example.

<br>

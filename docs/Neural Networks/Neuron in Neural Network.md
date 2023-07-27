## They're functions. The End.

<img src="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/47_blog_image_2.png" width="600">

## Brain and Machine

<img src="https://miro.medium.com/max/610/1*SJPacPhP4KDEB1AdhOFy_Q.png" width="600">

In a neural network, we try to simulate how your brain works by using lots of artificial neurons that are connected to each other.

Just like in your brain, these artificial neurons can turn on or off depending on the inputs they receive from other neurons.

When we connect lots of these neurons together in a network, we can use them to solve all sorts of stuff, like recognizing handwritten letters or predicting the weather.

## Inputs, Weights, and Bias

<img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/614fc05e2486109794ed3bdc_neuron.png" width="600">

A neuron is typically represented as a function.

```python
import numpy as np

def neuron(inputs, weights, bias):
    # Multiply the inputs by the weights
    weighted_inputs = np.dot(inputs, weights)
    # Add the bias to the weighted inputs
    weighted_sum = weighted_inputs + bias
    # Apply the activation function (in this case, a sigmoid function)
    output = 1 / (1 + np.exp(-weighted_sum))
    return output
```

<br>
In this example, the neuron function takes in three arguments:

1. inputs, which is an array of input values
2. weights, which is an array of weights that correspond to each input
3. bias, which is a single value that is added to the weighted sum of the inputs

The function then multiplies the inputs by the weights and adds the bias to get the weighted sum. It then applies an activation function (in this case, a sigmoid function) to the weighted sum to produce the output of the neuron.

<br>

## Activation Functions Summary

<span style="color:orange;font-size:larger;">Hidden layer activation = usually ReLU</span>

<span style="color:#0000dd;font-size:larger;">Output activation:</span>

<span style="color:aquamarine;font-size:larger;">Binary Classification = Sigmoid; Multiclass Classification = Softmax</span>

<span style="color:#0000dd;font-size:larger;">Loss function: Binary class = binary crossentropy; Multiclass class = crossentropy</span>

<span style="color:#00ffff;font-size:larger;">Optimizer: SGD and Adam</span>

Activation functions are like a gatekeeper for each neuron in a neural network. When the neuron receives input, the activation function determines whether or not the neuron should "fire" or activate.

In this particular problem of classifying images of digits, ReLU activation is often used in hidden layers due to its ability to speed up the training process and improve the performance of the neural network. Softmax activation is used in the output layer because it normalizes the output to a probability distribution over the classes, which is suitable for multi-class classification problems like MNIST.

## Softmax

**Softmax:** When we want a neural network to tell us what it thinks an input belongs to from a list of possibilities, we use the softmax function. It takes a list of numbers and turns them into a set of probabilities that **add up to 1.0**. This way, we can see which category the neural network thinks the input belongs to most.


**Softmax:** The softmax function is used for multi-class classification problems, where the goal is to assign an input to one of several possible classes. The output of the softmax function is a probability distribution over the classes, with each value representing the probability of the input belonging to that class. Its graph looks like a curve that starts at zero, rises steeply, and then levels off as it approaches 1.

<!--<img src="https://miro.medium.com/max/1838/1*670CdxchunD-yAuUWdI7Bw.png" width="768">-->

<img src="https://developers.google.com/static/machine-learning/crash-course/images/SoftmaxLayer.svg" width="600">

<img src="https://i.ytimg.com/vi/dmngpQv5D9Y/maxresdefault.jpg" width="768">

**Softmax** activation function is typically used in the output layer of neural networks when there are multiple classes to predict.

* Softmax converts a vector of real numbers into a probability distribution that sums to 1.0.
* The output of the softmax function can be interpreted as the probability of each class.

## Sigmoid

**Sigmoid:** This function helps us take any number and squish it between **0 and 1**, like squeezing toothpaste out of a tube. We use the sigmoid function when we want to know the probability of something happening. It's commonly used in binary classification problems, where we want to know whether something is either true or false.

**Sigmoid:** The sigmoid function is an S-shaped curve that maps any input value to a value between 0 and 1. Its graph looks like a stretched out "S" shape, with the output ranging from 0 at the left end to 1 at the right end.

<!--![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png)-->

<img src="https://ambrapaliaidata.blob.core.windows.net/ai-storage/articles/Untitled_design_13.png" width="600">

**Sigmoid** activation function is an S-shaped curve that maps any input value to a value between 0 and 1.

* It is commonly used in binary classification problems, where the goal is to predict whether an input belongs to one of two classes.
* The sigmoid function has the property that its output is always between 0 and 1, which makes it useful for predicting probabilities.

## tanh

**tanh:** Think of this as a function that takes a number as input and then squishes it to be between **-1 and 1**. We use this function when we want to make sure that the output of a neuron is always within a certain range. It's like putting a rubber band around the output to make sure it doesn't get too big or too small.

**tanh:** The shape of the hyperbolic tangent function is similar to that of the sigmoid function, but it ranges from -1 to 1. Its graph looks like a stretched out "S" shape, with the output ranging from -1 at the left end to 1 at the right end.

<!--![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Activation_tanh.svg/640px-Activation_tanh.svg.png)-->

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-27_at_4.23.22_PM_dcuMBJl.png">

**tanh (Hyperbolic Tangent)** activation function outputs values between -1 and 1.

* It is similar to the sigmoid function, but with a range that is twice as large.
* The output of the tanh function is always centered around zero, which makes it useful for normalizing input data.
* It is commonly used in hidden layers of neural networks.

## ReLU

**ReLU:** Imagine you have a light switch that only turns on if there is enough electricity flowing through it. This is what the ReLU function does. It only outputs a signal if the input is above a certain threshold, like turning on a light switch. This function is very fast to compute and is commonly used in deep neural networks.

**ReLU:** The ReLU function is a simple, linear function that outputs 0 for negative inputs and the input itself for positive inputs. Its graph is a straight line that passes through the origin, with a slope of 1 for positive values and 0 for negative values.

<!--![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/640px-Activation_rectified_linear.svg.png)-->

<img src="https://www.nomidl.com/wp-content/uploads/2022/04/image-10.png" width="600">

**ReLU** (Rectified Linear Unit) activation function sets **all negative values to zero**, while leaving positive values unchanged.

* ReLU is computationally efficient and has been shown to work well in many neural network architectures.
* It is commonly used in the hidden layers of deep neural networks.

<br>

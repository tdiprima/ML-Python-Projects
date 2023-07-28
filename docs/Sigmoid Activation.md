## Sigmoid Activation <span style="font-size: 27px;text-transform: lowercase;">σ</span>

<!--Greek small letter sigma σ-->

Sigmoid activation is a function that takes a number as input and squashes it to a value **between 0 and 1**.

So if you give it a really big number, it will output a number close to 1. 

If you give it a really small number, it will output a number close to 0. 

But if you give it a number in the middle, it will output a number somewhere in between 0 and 1, like 0.5.

The sigmoid activation function is often used in machine learning to "squash" the output of a neural network, so that it can be interpreted as a **probability**.

For example, if you're trying to predict whether an image contains a cat or a dog, you might use a neural network with a sigmoid activation function at the end to give you a number between 0 and 1 that represents the probability that the image contains a cat.

* High number = it's likely a cat
* Low number = it's likely a dog

<span style="color:#0000dd;font-size:larger;">But aren't all outputs between 0 and 1 anyway?</span>

Good point!

In many cases, the outputs of machine learning models are indeed already constrained to the range between 0 and 1, such as in the case of probability estimates.

But there are still situations where it can be useful to apply a sigmoid activation function.

One example is in binary classification tasks, where the goal is to predict one of two possible outcomes, such as "yes" or "no".

In such cases, the final output of the model can be interpreted as the probability of the positive outcome (e.g., **the probability of "yes"**).

By applying a sigmoid activation function to the final output, we can <mark>**ensure**</mark> that the resulting probability estimate is **always between 0 and 1.**

Another example is in cases where we want the model to have a nonlinear response to its inputs.

The sigmoid function is a nonlinear function that can introduce **curvature** into the model's decision boundary, which can help it learn more complex patterns in the data.

### Linear / Nonlinear

The sigmoid function is a nonlinear function, which means that it can introduce curvature into the output.

This curvature can help the model learn more complex patterns in the data that a linear function (like multiplication by a constant) might not be able to capture.

<br>
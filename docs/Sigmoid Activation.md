## Sigmoid Activation <span style="font-size: 27px;text-transform: lowercase;">σ</span>

Greek small letter **sigma σ**

Sigmoid activation is a function that takes a **number** as input and **squashes it to a value between 0 and 1.** 

So if you give it a **really big** number, it will output a number close to **1**. 

If you give it a **really small** number, it will output a number close to **0**. 

But if you give it a **number in the middle,** it will output a number somewhere in between 0 and 1, like **0.5**.

<mark>The sigmoid activation function is often used in machine learning to **"squash" the output of a neural network,** so that it can be interpreted as a **probability.**</mark>

For example, if you're trying to predict whether an image contains a **cat or a dog,** you might use a neural network with a sigmoid activation function at the end to give you a number **between 0 and 1** that represents the **probability that the image contains a cat**.

* High number = it's likely a cat
* Low number = it's likely a dog

### But aren't all outputs between 0 and 1 anyway?

**That's a good point!**

In many cases, the outputs of machine learning models are indeed already constrained to the **range between 0 and 1,** such as in the case of **probability estimates.**

However, there are still situations where it can be useful to apply a sigmoid activation function.

One example is in **binary classification tasks,** where the goal is to predict one of two possible outcomes, such as "yes" or "no".

In such cases, the final output of the model can be interpreted as the probability of the positive outcome **(e.g., the probability of "yes").**

By **applying a sigmoid activation function to the final output,** we can ensure that the resulting probability estimate is **always between 0 and 1.**

Another example is in cases where we want the model to have a **nonlinear response to its inputs.**

The sigmoid function is a nonlinear function that can **introduce curvature** into the model's decision boundary, which can help it learn more complex patterns in the data. <!-- TODO -->

So while not all machine learning models require sigmoid activation, it can be a useful tool in certain situations.

### So... it's kind of like when we multiply a number by 0.5 in math?

**That's a good analogy!**

When you multiply a number by 0.5, you're essentially **scaling it down by a factor of 2.** 

Similarly, <mark>**when you apply a sigmoid function to a number, you're scaling it down to a value between 0 and 1.**<mark>

But there is one important difference: the sigmoid function is a nonlinear function, which means that it can introduce curvature into the output. This curvature can help the model learn more complex patterns in the data that a linear function (like multiplication by a constant) might not be able to capture.

So while the idea of scaling a number down to a value between 0 and 1 is similar, the sigmoid function has some additional properties that make it useful for machine learning.

## "Factor Of"

When we talk about scaling something by a factor, we mean that we're multiplying it by a certain amount. For example, if we say we're scaling something by a factor of 2, that means we're multiplying it by 2.

Let's take the example of multiplying a number by 0.5, like you mentioned earlier. When we multiply a number by 0.5, we're essentially scaling it down by a factor of 2. Why? Because 0.5 is one-half, or one divided by two. So if we multiply a number by 0.5, we're essentially dividing it by 2.

For example, if we take the number 10 and multiply it by 0.5, we get:

<span style="color: maroon; font-weight:bold; font-size: larger">10 x 0.5 = 5</span>

So we've scaled the number down by a factor of 2 **(i.e., we've divided it by 2).**

<br>

<span style="color: #000088; font-size: larger">OK, so that means... we can say "factor of 2" when we multiply by 2, or multiply by 0.5, because each result has to do with the number 2.</span>

<span style="color: #000088; font-size: larger">Like "double" is "2", in the case of multiplying by 2.</span>

<span style="color: #000088; font-size: larger">And multiplying by 0.5 is a factor of 2, because we are essentially dividing by 2, when we do that.</span>

**Yes, exactly!** When we say "factor of 2", we mean that we're multiplying by 2 or dividing by 1/2 (which is the same thing as multiplying by 2). So if we multiply a number by 2, we've increased it by a factor of 2 (i.e., we've doubled it). And if we multiply a number by 0.5, we've decreased it by a factor of 2 (i.e., we've halved it).

The terms "double" and "halve" are often used to describe multiplying by 2 and dividing by 2, respectively.

<mark>**So when we say "scaling by a factor of 2", we're essentially saying "doubling" or "halving" the number, depending on whether we're multiplying by 2 or dividing by 2.**</mark>

<span style="font-size: 27px;">☀️</span> I hope that clears things up!

## Multiply by 0.1.

When we multiply a number by 0.1, it means we're finding 1/10th of that number.

For example, if we have the number 50 and we want to multiply it by 0.1, we would do:

<span style="color: maroon; font-weight:bold; font-size: larger">50 x 0.1 = 5</span>

So, when we multiply a number by 0.1, we're essentially taking 10% of that number. In other words, we're finding what number is equivalent to one-tenth of the original number.

### Ok, now...

I was gonna ask about application, like in three, but.

## greek capital letter sigma Σ

Absolutely! The Greek capital letter sigma (Σ) is a symbol used in math to represent the idea of **"adding things up".**

When you see the sigma symbol in a math equation, it's telling you to **add up a series of numbers** or terms. For example, if you saw an equation like this:

Σn

It means you need to add up all the **numbers from 1 to n**. So if n was 5, you would add 1 + 2 + 3 + 4 + 5.

Another example could be:

Σx_i

Here, it means you need to add up all the values of x_i.

The "i" is just a way of labeling each value of x so you can keep track of them.

So, in short, the sigma symbol is a shorthand way of saying <mark>**"add up all of these things".**</mark>

### What that means? <span style="font-size: 25px;text-transform:lowercase;">x_i</span>

Let's say you have a set of numbers: 2, 4, 6, 8, and 10. You can represent each number in the set as x1, x2, x3, x4, and x5 respectively. So:

* x1 = 2
* x2 = 4
* x3 = 6
* x4 = 8
* x5 = 10

Now, if you see the equation Σx\_i, it means you need to add up all the values of x\_i in the set. In this case, that would be:

```
Σx_i = x1 + x2 + x3 + x4 + x5
Σx_i = 2 + 4 + 6 + 8 + 10
Σx_i = 30
```

<br>
So, Σx_i in this case would be equal to 30.

### How do you pronounce "Σx_i" in words?

You can pronounce Σx_i as "the sum of x sub i" or "the sum of x with subscript i". The subscript i is just a way of distinguishing each element in the sequence or set of values that you're summing up.

<br>

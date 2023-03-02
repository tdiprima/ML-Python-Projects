## Optimization algorithm called Adam 🍎

Adam, which is a popular choice for training neural networks.

Adam stands for "Adaptive Moment Estimation", and it is a variant of stochastic gradient descent (SGD) that adapts the learning rate based on the first and second moments of the gradients.

Adam is known for its fast convergence and robustness to noisy gradients, and it is often used as the default optimizer in many machine learning frameworks, including TensorFlow and Keras.

Optimization algorithms used in machine learning:

* Gradient Descent
* Adagrad
* RMSprop
* Etc.


<!--## Adam optimizer 🦸🏻‍♂️-->

## Explain

```py
# Train the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
```

<br>
When we train a machine learning model, we want to adjust the parameters of the model so that it can make accurate predictions on new data.

**The process of adjusting the parameters is called optimization.**

There are many optimization algorithms, and one of the most popular is called the Adam optimizer.

The Adam optimizer is an algorithm that adjusts the parameters of a model in order to minimize the error between the model's predictions and the actual values.

The basic idea behind Adam is to **adjust the learning rate** of the optimization algorithm based on the gradient of the loss function.

(Remember? Based on GSD.)

The **learning rate** determines how quickly the algorithm should adjust the parameters of the model in response to the error.

If the learning rate is too high, the algorithm may overshoot the optimal values and fail to converge.

If the learning rate is too low, the algorithm may converge too slowly or get stuck in local optima.

The Adam optimizer also includes an **adaptive momentum** term that adjusts the learning rate based on the history of the gradients.

This helps the optimizer to navigate narrow ravines and saddle points in the loss function.

By adapting the learning rate based on the gradient history, the Adam optimizer can converge more quickly and more reliably than other optimization algorithms.

### Analogy 🧸

Imagine you are playing a game of "Hot and Cold" with a friend. Your friend has hidden a toy in a room, and your job is to find it by moving around the room. Your friend will give you clues by telling you whether you are getting "hotter" or "colder" as you move around. If you are getting closer to the toy, your friend will say "hotter". If you are moving away from the toy, your friend will say "colder".

The Adam optimizer works like this game of "Hot and Cold". The optimizer starts with a random guess for the parameters of the model, and calculates the gradient of the loss function with respect to those parameters.

If the gradient is "hotter" (larger), the optimizer will take a bigger step in that direction.

If the gradient is "colder" (smaller), the optimizer will take a smaller step.

*Sounds like reinforcement learning to me.*

By **adjusting the learning rate** based on the gradient, the Adam optimizer can quickly find the optimal values of the parameters and make accurate predictions on new data.

<br>

## Neural Network Optimizers

<span style="color:#0000dd;">The optimizer of a neural network helps to adjust weights and gradient descent, right?</span>

Yes, that's correct! The optimizer of a neural network helps **adjust the weights** of the network in order **to minimize the loss function** during the training process. The weights are initialized randomly at the start of training, and during each training iteration, the optimizer updates the weights based on the gradient of the loss function with respect to the weights.

```ruby
# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Gradient descent** is one of the most common optimization algorithms used in neural networks. It works by **computing the gradient** of the loss function with respect to the weights at each training step, and then **updating the weights in the opposite direction** of the gradient in order to minimize the loss.

There are several variants of gradient descent, including stochastic gradient descent **(SGD)**, which randomly selects a subset of training examples for each update step, and adaptive algorithms like **Adam** or **RMSprop** which can adjust the learning rate based on the gradient magnitude.

The optimizer is a crucial component of the neural network training process, as it determines how quickly and effectively the network can learn from the training data. Choosing the right optimizer and setting its hyperparameters can be important for achieving good performance on a given task.

## Adam optimizer

Adam, which is a popular choice for training neural networks.

Adam stands for "Adaptive Moment Estimation", and it is a variant of stochastic gradient descent (SGD) that adapts the learning rate based on the first and second moments of the gradients.

Adam is known for its fast convergence and robustness to noisy gradients, and it is often used as the default optimizer in many machine learning frameworks, including TensorFlow and Keras.

Optimization algorithms used in machine learning:

* Gradient Descent
* Adagrad
* RMSprop
* Etc.


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

## Machine learning algorithms 🤖

* Linear regression
* Logistic regression
* Decision trees
* Random forests
* Support vector machines
* K-nearest neighbors
* Neural networks
* Etc.

*It's worth noting that some machine learning algorithms use parameters that are set to a value of 50 or multiples of 50.*

*For example, the k-nearest neighbors algorithm uses a parameter `k` that specifies the number of nearest neighbors to consider when making predictions, and this value could be set to 50 or a multiple of 50.*

*However, the value of `k` is typically chosen based on cross-validation or other model selection techniques, rather than being set to a fixed value.*

<br>

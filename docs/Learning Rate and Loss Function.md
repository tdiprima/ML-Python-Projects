## Loss Function

In machine learning, we use a **"loss function"** to measure how well our model is doing at making predictions. The loss function takes in the predicted output of the model and the true output (i.e. the "ground truth") and returns a number that represents how well the model is doing.

The goal of the learning process is to **minimize the loss function**, which means we want to make the model's predictions as close as possible to the true outputs. To do this, we use an optimization algorithm called "gradient descent" to update the model's parameters (i.e. the weights and biases of the neurons) in the direction that reduces the loss.

```ruby
# Create loss function
loss_fn = nn.L1Loss()

# Calculate difference between 
# model's predictions on training set,
# and the ideal training values.
loss = loss_fn(y_pred, y_train)
```

## Learning Rate

The **"learning rate"** is a parameter that determines **how much** the model's parameters are updated during each iteration of the learning process. A higher learning rate means that the model's parameters will be updated more aggressively, which can help the model converge to a good solution faster. However, if the learning rate is too high, the model may overshoot the optimal solution and diverge.

A **lower learning rate** means that the model's parameters will be updated more slowly, which can help the model avoid overshooting the optimal solution.

However, if the learning rate is too low, the model may take a long time to converge to a good solution, or it may get stuck in a suboptimal solution.

**Choosing a good learning rate** is an important part of training a machine learning model. In practice, you usually start with a reasonable learning rate (such as 0.01) and experiment with adjusting it to see how it affects the performance of the model.

```ruby
# Create optimizer
optimizer = torch.optim.SGD(
    params=model_0.parameters(), 
    lr=0.01
)
```

## Bonus!

For a **regression** problem, use:

* Loss function `nn.L1Loss()`
    * Same as MAE (mean absolute error)
* Optimizer `torch.optim.SGD()`

For a **classification problem**, use:

* Loss function `nn.BCELoss()`
    * (binary cross entropy loss)

<table border="1">
<tr>
<th>Loss function/Optimizer</th>
<th>Problem type</th>
<th>PyTorch Code</th>
</tr>
<tr>
<td>Stochastic Gradient Descent (SGD) optimizer</td>
<td>Classification, regression, etc.</td>
<td>torch.optim.SGD()</td>
</tr>
<tr>
<td>Adam Optimizer</td>
<td>Classification, regression, etc.</td>
<td>torch.optim.Adam()</td>
</tr>
<tr>
<td>Binary cross entropy loss</td>
<td>Binary classification</td>
<td>torch.nn.BCELossWithLogits or torch.nn.BCELoss</td>
</tr>
<tr>
<td>Cross entropy loss</td>
<td>Multi-class classification</td>
<td>torch.nn.CrossEntropyLoss</td>
</tr>
<tr>
<td>Mean absolute error (MAE) or L1 Loss</td>
<td>Regression</td>
<td>torch.nn.L1Loss</td>
</tr>
<tr>
<td>Mean squared error (MSE) or L2</td>
<td>Regression</td>
<td>torch.nn.MSELoss</td>
</tr>
</table>
<!-- DivTable.com -->

<br>

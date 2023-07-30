## Loss Function

```py
loss_fn = nn.CrossEntropyLoss()
```

<br>
Loss function is often called "criterion"; the criterion you're trying to reduce.

In machine learning, we use a **"loss function"** to measure how well our model is doing at making predictions. The loss function takes in the predicted output of the model and the true output (i.e. the "ground truth") and returns a number that represents how well the model is doing.

The goal of the learning process is to **minimize** the loss function, which means we want to make the model's predictions as close as possible to the true outputs.

To do this, we use an **optimization** algorithm called **"gradient descent"** to update the model's parameters (i.e. the weights and biases of the neurons) in the direction that reduces the loss.

```python
import torch
import torch.nn as nn

# Pretend we made some predictions
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)

# Compute the average loss
mae_loss = nn.L1Loss()
output = mae_loss(input, target)
output.backward()

print('\nInput: ', input)
print('\nTarget: ', target)
print('\nOutput: ', output)
```

<br>
The Mean Absolute Error (MAE), also called L1 Loss, computes the average of the sum of absolute differences between actual values and predicted values.

[neptune.ai](https://neptune.ai/blog/pytorch-loss-functions)

Used for **regression problems**, especially when the distribution of the target variable has outliers, such as small or big values that are a great distance from the mean value. It is considered to be more robust to outliers.

## Learning Rate

The **"learning rate"** is a parameter that determines **how much** the model's parameters are updated during each iteration of the learning process. A higher learning rate means that the model's parameters will be updated more aggressively, which can help the model converge to a good solution faster. However, if the learning rate is too high, the model may overshoot the optimal solution and diverge.

A **lower learning rate** means that the model's parameters will be updated more slowly, which can help the model avoid overshooting the optimal solution.

However, if the learning rate is too low, the model may take a long time to converge to a good solution, or it may get stuck in a suboptimal solution.

**Choosing a good learning rate** is an important part of training a machine learning model. In practice, you usually start with a reasonable learning rate (such as **`0.01`**) and experiment with adjusting it to see how it affects the performance of the model.

## Optimization

SGD and Adam are 2 of the most popular optimizers.

[How to optimize a function using SGD](https://www.projectpro.io/recipes/optimize-function-sgd-pytorch)

```python
import torch

# Define parameters
batch_size, dim_input, dim_hidden, dim_output = 64, 1000, 100, 10

# Create Random tensors
input_X = torch.randn(batch_size, dim_input)
output_Y = torch.randn(batch_size, dim_output)

# Define model and loss function
SGD_model = torch.nn.Sequential(torch.nn.Linear(dim_input, dim_hidden), torch.nn.ReLU(), torch.nn.Linear(dim_hidden, dim_output))

loss_fn = torch.nn.MSELoss(reduction='sum')

# Define learning rate
rate_learning = 0.1

# Initialize optimizer (to update the weights of the model for us)
optim = torch.optim.SGD(
    SGD_model.parameters(),  # The parameter's we're gonna optimize
    lr=rate_learning,
    momentum=0.9
)

# Forward pass
for values in range(500):
    # Predict y by passing input_X to the model
    pred_y = SGD_model(input_X)

    # Compute the loss
    loss = loss_fn(pred_y, output_Y)

    if values % 100 == 99:
        print(values, loss.item())
```

## BCELoss + Adam for Binary Classification

<span style="color:#0000dd;">For a **regression** problem, use: `nn.L1Loss()` and `torch.optim.SGD()`.</span>

<span style="color:#0000dd;">For a **classification** problem, use: `nn.BCELoss()` and `torch.optim.Adam()` optimizer.</span>

Adam (Adaptive Moment Estimation) is a popular optimizer that is well-suited for a wide range of machine learning tasks, including classification. It adapts the learning rates of each parameter based on the past gradients, making it efficient in converging to a good solution in many scenarios.

Here's an example of how you can set up the optimizer for a binary classification problem using `nn.BCELoss()`:

```python
import torch.nn as nn
import torch.optim as optim


# Assuming you have defined your model and the loss function as follows
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define your model layers here

    def forward(self, x):
        # Implement the forward pass of your model here
        return x


model = YourModel()
criterion = nn.BCELoss()

try:
    # You can adjust the learning rate (lr) as needed
    optimizer = optim.Adam(model.parameters(), lr=0.001)
except Exception as ex:
    print(ex)
```

<br>

Remember to adjust the learning rate (`lr`) based on your specific problem and dataset. You may need to tune it to achieve the best performance.

If your classification problem is multi-class instead of binary, you should use a different loss function like `nn.CrossEntropyLoss()` or `nn.NLLLoss()` (Negative Log Likelihood Loss), depending on whether your model's final layer includes a softmax activation or not.

In such cases, the choice of optimizer can still be `torch.optim.Adam()` or other optimizers like `torch.optim.SGD()`, depending on the specific requirements of your problem.

<br>
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

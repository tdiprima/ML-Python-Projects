## Running PyTorch code on a GPU

<span style="color:red;">The Model and the Tensors get put on the device.</span>

The End.

```python
# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Put model on target device
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

# Put data on target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

```

<br>

> "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"

I thought I was specifying gpu/cuda everywhere.  What should I be checking, in order to stop these errors?

The error message "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!" indicates that you have tensors on different devices (CPU and GPU).

This can happen if you forget to **move a tensor** to the GPU or if you accidentally move it back to the CPU.

<span style="color:#0000dd;font-weight:bold;font-size:larger;">Get it?</span>  <span style="color:#0000dd;font-size:larger;">"Move..."  To!</span>

```py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ...
dummy_x = torch.rand([1, 1, 28, 28]).to(device)
```

<br>

To fix this error, you should ensure that **all your tensors are on the same device** (i.e., either on the CPU or GPU).

1. Make sure that you're using the correct device throughout your code.
    * You can specify the device by using the `.to()` method.
    * For example, if you have a tensor `x` and you want to move it to the GPU, you can do `x = x.to('cuda:0')`.

2. Check that you're not inadvertently creating new tensors on the CPU.
    * For example, if you're using functions that create new tensors, make sure to move the output tensors to the same device as your input tensors.

3. Check that you're not inadvertently moving tensors back to the CPU.
    * For example, if you're using functions that return tensors on the CPU, make sure to move them to the same device as your other tensors.

4. Check that you're not inadvertently using operations that only work on CPU tensors.
    * For example, some PyTorch operations (e.g., string operations) only work on CPU tensors.
    * Make sure to avoid using these operations on GPU tensors.


## Identify and fix the issues that are causing tensors to be on different devices

I moved my model to the GPU using `.to("cuda:0")`.

But it's still throwing the "same device" error when I do the forward pass: `y_pred = model_0(X)`.

`X` is simply instantiated on the fly when we say: `for batch, (X, y) in enumerate(train_dataloader):`...

And since you can't do `.to("cuda:0")` on a `DataLoader`, I don't understand why I'm still getting this error.

<span style="color:#0000dd;font-weight:bold;font-size:larger;">That's it.</span> <span style="color:#0000dd;font-size:larger;">You have to do X.to(device), and y.to(device)</span>

The error message suggests that some of the tensors in your forward pass are still on the CPU.

As you've correctly noted, you cannot call `.to("cuda:0")` directly on the `DataLoader` object **because it is an iterable, not a tensor.**

However, you can move the batches of data to the GPU by calling `.to("cuda:0")` **on each batch tensor** in the `for` loop.

```scss
for batch, (X, y) in enumerate(train_dataloader):
    X = X.to("cuda:0")
    y = y.to("cuda:0")

    # perform forward pass
    y_pred = model_0(X)

    # rest of training loop...
```

This will ensure that both `X` and `y` are on the same device (in this case, the GPU) as the model's parameters, which should eliminate the "same device" error you're seeing.

<br>

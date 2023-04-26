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

## Identify and fix the issues that are causing tensors to be on different devices

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

## Each batch tensor in for-loop

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

## torch.cuda.amp.autocast()

You can think of CPU and GPU like a calculator; a calculator that the computer uses.

And just like how some calculators are faster or more powerful than others, some parts of your computer are better at doing certain kinds of math than others.

For example, the graphics processing unit, or **GPU** for short. It's kind of like a really fast calculator that's designed specifically for doing a certain kind of math called **matrix multiplication.**

So when you're using a library like PyTorch to do machine learning, sometimes you'll want to use your GPU to speed things up. And that's where `torch.cuda.amp.autocast()` comes in.

What this function does is **it tells PyTorch to use your GPU** to do the math for you, but it also helps make sure that the math is done as accurately as possible. It's like having a really smart calculator that not only works really fast, but also helps you avoid making mistakes.

So in short, `torch.cuda.amp.autocast()` is a way to use your computer's GPU to do math really quickly and accurately when you're doing machine learning with PyTorch.

## torch.cuda.amp.GradScaler()

When you're training a machine learning model with PyTorch, you'll often need to **adjust the weights of the model based on how well it's doing** on the training data.

This process is called **backpropagation**, and it involves **calculating gradients** to figure out which direction to adjust the weights in.

Now, just like with `torch.cuda.amp.autocast()`, using your computer's GPU to do these calculations can make things go a lot faster. **But there's one problem:** sometimes the gradients can get really big or really small, and that can cause problems with the **accuracy** of the calculations.

That's where `torch.cuda.amp.GradScaler()` comes in. This function helps **make sure that the gradients stay within a certain range,** so that the calculations stay accurate even when using the GPU. It does this by **scaling the gradients** up or down before applying them to the model weights.

Think of it like this: imagine you have a bucket of water and you need to pour it into a smaller container. If you try to pour it all at once, you might end up spilling some of the water because the smaller container can't handle it all. But if you pour a little bit at a time, you can make sure that none of it spills over the edge.

`torch.cuda.amp.GradScaler()` is kind of like pouring the gradients into the model weights a little bit at a time, to make sure that nothing "spills over" and causes inaccuracies.

So in summary, `torch.cuda.amp.GradScaler()` is a way to help make sure that the gradients used in backpropagation during model training stay within a certain range, even when using the GPU to speed things up.

<br>

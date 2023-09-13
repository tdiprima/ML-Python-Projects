## Typical PyTorch program

<a href="../misc/typical-pytorch.py">pgm</a>

1. Import the necessary libraries, then define some **hyperparameters:**
    * Learning rate
    * Number of epochs
    * Batch size

2. **Load** and preprocess data.

3. Define our model **architecture** using PyTorch's `nn.Sequential` API.

4. Define our **loss function** and **optimizer** and set the model's parameters to be optimized.

5. **Train** our model using a loop that goes through each epoch and each batch of data.
    6. We first zero out the gradients
    7. Perform a forward pass through the model
    8. Compute the loss
    9. Perform a backward pass
    10. Update the model's weights

6. **Evaluate** our model on test data using the `with torch.no_grad()` context manager to turn off gradient computation. We also make some predictions with the model.

7. **Print out some results**:
    * Test accuracy
    * Predictions

## MNIST classification

*Because Buddy omitted the load\_and\_preprocess\_data() code...*

For what dataset might you define your PyTorch model architecture like the following?

```python
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

<br>

This architecture seems to be suitable for a **classification** task on the MNIST dataset, which is a collection of **28x28** grayscale images of handwritten digits **(0 to 9).**

```c
28 * 28 = 784
```

The input size of **784** corresponds to flattening the 28x28 images into a 1D tensor of length 784.

The output size of **10** represents the number of classes (digits) in the dataset.

The **ReLU** activation function is commonly used in neural networks for classification tasks.

The choice of **64** hidden units in the middle layer is a common choice for a simple and effective architecture.

<br>

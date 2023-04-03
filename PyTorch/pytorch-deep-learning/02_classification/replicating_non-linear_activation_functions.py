import torch
from matplotlib import pyplot as plt

# Replicating non-linear activation functions
A = torch.arange(-10, 10, 1, dtype=torch.float32)
print("A.dtype", A.dtype)
print("A", A)

# Visualize the tensor
plt.plot(A)
plt.show()


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x)  # inputs must be tensors


print("relu(A)", relu(A))

# Plot ReLU activation function
# plt.plot(torch.relu(A))  # See? It's the same thing.
plt.plot(relu(A))
plt.show()


# Now let's do the same for Sigmoid
# https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# print("sigmoid(A)", sigmoid(A))

# plt.plot(torch.sigmoid(A))  # Same.
plt.plot(sigmoid(A))
plt.show()

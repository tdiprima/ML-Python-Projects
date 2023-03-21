import torch
import torch.nn as nn

# Define your model output and ground truth
logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]])
labels = torch.tensor([0, 1, 2])

# Create a CrossEntropyLoss object
loss_fn = nn.CrossEntropyLoss()

# Calculate the loss
loss = loss_fn(logits, labels)

print(loss.item())

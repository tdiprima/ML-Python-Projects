## Calculate accuracy

```ruby
# Setup test loss and test accuracy values
test_loss, test_acc = 0, 0

# Send data to the target device
X, y = X.to(device), y.to(device)

# Forward pass
test_pred_logits = model(X)

# Calculate the accuracy
test_pred_labels = test_pred_logits.argmax(dim=1)
# ¿Qué es esto?
test_acc += (
    (test_pred_labels == y).sum().item() / len(test_pred_labels)
)
```

<br>

<a href="../pytorch-deep-learning/04_custom_datasets/test_train_def.py">test\_train\_def.py</a>

## Counting True's and False's

`test_pred_labels == y` returns a `torch.Tensor` containing True's and False's.

`(test_pred_labels == y).sum()` returns a tensor containing one number in it, which represents the **count of the number of `True` values** in the boolean tensor resulting from the comparison `test_pred_labels == y`.


## Like this one

```ruby
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc
```

<a href="../pytorch-deep-learning/02_classification/05.py">05.py</a>

<br>

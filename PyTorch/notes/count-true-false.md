## Calculate accuracy

```ruby
test_pred_labels = test_pred_logits.argmax(dim=1)
test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
```

<br>

<span style="color:#0000dd;">pytorch-deep-learning/04\_custom\_datasets/test\_train\_def.py</span>

If `test_pred_labels == y` returns a `torch.Tensor` containing True's and False's,
what does `(test_pred_labels == y).sum()` return? A tensor containing one number in it?  e.g. `tensor(19)`.

## Counting True's and False's

Yes, `(test_pred_labels == y).sum()` returns a tensor containing a **single number,** which represents the **count of the number of `True` values** in the boolean tensor resulting from the comparison `test_pred_labels == y`.

The `==` operator performs an element-wise comparison between `test_pred_labels` and `y`, resulting in a boolean tensor with the same shape as the input tensors.

Each element of the resulting tensor is `True` if the corresponding element of `test_pred_labels` and `y` are equal, and `False` otherwise.

The `.sum()` method then sums up all the `True` values in the boolean tensor, returning a tensor containing a single number that represents the **count of the `True` values.**

## Like this one

```ruby
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc
```

<span style="color:#0000dd;">pytorch-deep-learning/02_classification/05.py</span>

<br>

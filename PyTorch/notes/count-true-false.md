## Calculate accuracy

```py
test_pred_labels = test_pred_logits.argmax(dim=1)
test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
```

<br>

If `test_pred_labels == y` returns a `torch.Tensor` containing True's and False's,
what does `(test_pred_labels == y).sum()` return &ndash; a tensor containing one number in it?  eg. `tensor(19)`

### Counting True's and False's

Yes, `(test_pred_labels == y).sum()` returns a tensor containing a single number, which represents the count of the number of `True` values in the boolean tensor resulting from the comparison `test_pred_labels == y`.

The `==` operator performs an **element-wise comparison** between `test_pred_labels` and `y`, resulting in a boolean tensor with the same shape as the input tensors.

Each element of the resulting tensor is `True` if the corresponding element of `test_pred_labels` and `y` are equal, and `False` otherwise.

The `.sum()` method then sums up all the `True` values in the boolean tensor, returning a tensor containing a single number that represents the **count of the `True` values.**

### Like this one

```python
correct = torch.eq(y_true, y_pred).sum().item()
acc = (correct / len(y_pred)) * 100
```

<br>

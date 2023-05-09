## Mnemonic for Training

What's a good way of remembering the order in which we train the model?

```ruby
# Training / Optimization
for epoch in range(epochs):
    model.train()

    y_pred = model(X_train)
    
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

<br>

Actually it's like "lolo". Lol!

Loss, optimizer, loss, optimizer.

And before that, we train and predict (train mode, that is).

One possible mnemonic device for remembering the order in which the model is trained using the code provided could be **"FLOZP"**, which stands for:

"Forward pass, Loss, Optimizer zero grad, Loss backwards, Progress optimizer".

## Steps:

* **Forward pass:** the input data is passed through the model to generate predictions.
* **Loss:** the difference between the predictions and the true labels is calculated.
* **Optimizer zero grad:** the gradients of the optimizer are reset to zero.
* **Loss backwards:** the gradients of the loss are calculated with respect to the model's parameters using backpropagation.
* **Progress optimizer:** the optimizer uses the gradients to update the model's parameters and improve its performance.

Using a memorable phrase like **"FLOZP"** can help you remember the sequence of steps more easily, especially if you repeat it to yourself several times while you're learning the process.

* Training mode
* Forward pass
* Loss
* Optimizer zero grad
* Loss backwards *ZZ*
* Progress optimizer (or the "P" in "step")

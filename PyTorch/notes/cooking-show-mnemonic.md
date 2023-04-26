## Mnemonic for Training

What's a good way of remembering the order in which we train the model?  How about a mnemonic device?

```ruby
# Training / Optimization
for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(X_train)
    
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

<br>

Actually it's like "lolo". Lol! 😂 Loss, optimizer, loss, optimizer.  And before that, we train and predict (train mode, that is).

One possible mnemonic device for remembering the order in which the model is trained using the code provided could be "FLOZP", which stands for **"Forward pass, Loss, Optimizer zero grad, Loss backwards, Progress optimizer".**

### Steps:

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


# Torch Flow

1. Get data ready
2. Build a model
    * Pick loss function & optimizer
    * Training loop 
3. Fit model to data (training)
4. Make predictions and evaluate model
5. Save model

## Analogy using cooking steps <span style="font-size:27px;">👩‍🍳</span>

1. Get ingredients ready (similar to getting data ready)
2. Develop a recipe (similar to building a model)
    * Spices (cost, optimizer, forward)
3. Cook the dish according to the recipe (similar to fitting the model to the data)
4. Taste the dish and adjust seasoning as necessary (similar to making predictions and evaluating the model)
5. Store the leftover dish for later (similar to saving the model)

<br>

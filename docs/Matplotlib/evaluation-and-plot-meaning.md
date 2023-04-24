## Evaluate TensorFlow after training.

<span style="color:#0000dd;">I've seen people evaluate their tensorflow model before training it.  Is this advisable?  Wouldn't we want to evaluate it after training?</span>

It is not advisable to evaluate a TensorFlow model before training it because the **weights** of the model are randomly initialized at the beginning of the training process, and the model has not yet **learned** anything from the training data.

During training, the model learns from the data and updates its weights to improve its performance on the task at hand. Therefore, evaluating the model before training would give you random predictions that are not representative of the model's actual performance.

It is important to evaluate the model after training to assess its performance and determine if it is meeting the desired accuracy and other metrics. This evaluation will help you fine-tune the model and improve its performance if necessary.

In summary, it is advisable to evaluate the TensorFlow model after training it, not before.


## Plot validation and training loss

<span style="color:#0000dd;">Let's say I used matplotlib to plot the validation loss and training loss.  What does it mean when the validation loss goes down, but then after 5 epochs it goes back up?</span>

If the validation loss initially decreases and then starts to increase after a certain number of epochs, **it is a sign of overfitting.**

Overfitting occurs when a model becomes too complex and starts to memorize the training data instead of learning the underlying patterns that generalize to new, unseen data.

When the model overfits, it performs well on the training data but poorly on the validation data. The validation loss initially decreases because the model is learning the underlying patterns in the training data. However, after a certain point, the model starts to memorize the training data and becomes too specialized to generalize well to new data, causing the validation loss to start increasing.

To address this issue, you can try the following techniques:

1. Use regularization techniques such as dropout or L1/L2 regularization to reduce the complexity of the model and prevent overfitting.

2. Use early stopping to stop the training process when the validation loss stops improving.

3. Increase the size of the validation set to get a better estimate of the model's performance on unseen data.

4. Reduce the learning rate of the optimizer to slow down the rate at which the model learns from the training data.

5. Try different architectures or hyperparameters to find the optimal configuration that balances the model's complexity and performance.

In summary, if the validation loss starts to increase after a certain number of epochs, it is a sign of overfitting, and you need to take steps to address this issue to improve the model's performance on new data.

## Plot accuracy

<span style="color:#0000dd;">OK.  And if the training loss goes down and stabilizes, at zero, I understand that that's a good thing.  But if the validation loss goes down then back up (overfitting), but the training loss goes down like it's supposed to, then... what does that mean?</span>

If the training loss goes down and stabilizes at zero while the validation loss initially decreases and then starts to increase, it is a sign that the model is overfitting to the training data.

When the training loss goes down to zero, it means that the model has learned to perfectly fit the training data, which is good. However, this does not necessarily mean that the model can generalize well to new, unseen data. If the validation loss starts to increase, it means that the model is starting to memorize the training data instead of learning the underlying patterns that generalize to new data, causing the model to overfit.

To address this issue, you can try the techniques mentioned earlier, such as regularization, early stopping, and hyperparameter tuning. You may also want to consider increasing the size of the validation set or using cross-validation to get a better estimate of the model's performance on unseen data.

In summary, if the training loss goes down to zero while the validation loss increases, it is a sign of overfitting, and you need to take steps to address this issue to improve the model's ability to generalize to new data.

## Plot accuracy

<span style="color:#0000dd;">So I think that means... when we plot training and validation accuracy, the validation line should preferably be above the training line, right?</span>

No. In fact, it is quite common for the **training accuracy to be higher than the validation accuracy** because the model is optimized to fit the training data. 🤦🏻‍♀️ <span style="color:#0000dd;">Duh.</span>

The goal of training a machine learning model is to achieve good generalization performance on new, unseen data. Therefore, the validation accuracy is a more important metric to monitor than the training accuracy.

Ideally, both the training and validation accuracy should increase as the model is trained, but the rate of increase in the validation accuracy may slow down or plateau at some point. This is because the model may start to overfit the training data as it becomes more complex, causing the validation accuracy to stop improving.

In summary, while it is not necessary for the validation accuracy to be above the training accuracy, **it is more important to focus on improving the validation accuracy** to achieve good generalization performance on new data.

<br>

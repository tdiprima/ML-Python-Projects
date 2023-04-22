## How do I get tensorflow to not print messages of type INFO?

You can set the logging level for TensorFlow to a higher level to avoid printing messages of type INFO. 

Here's an example code snippet that sets the logging level to WARNING:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2', '3'}

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
```

In the code snippet above, we set the environment variable `TF_CPP_MIN_LOG_LEVEL` to `'2'` which sets the TensorFlow logging level to `WARNING` and disables the messages of type `INFO`. Additionally, we set the logging verbosity to `ERROR` using `tf.compat.v1.logging.set_verbosity` to avoid printing any further warning or error messages.

You can change the value of `TF_CPP_MIN_LOG_LEVEL` to any of the following:

- 0: Display all messages (default behavior).
- 1: Filter out INFO messages.
- 2: Filter out INFO and WARNING messages.
- 3: Filter out all messages except for ERROR messages.

Note that this will affect all TensorFlow logging messages in your script.

If you only want to suppress the INFO messages **for a particular section** of code, you can use the following context manager:

```python
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
``` 

This will suppress the INFO messages within the context of the `with` block.

## Gweat.

<span style="color:#0000dd;">Now... when you run classifier.train(), it prints a bunch of INFO, but only the last message is important: Loss for final step: 0.4708705.</span>

<span style="color:#0000dd;font-weight:bold;">Is there a way to just get that last message?</span>

Yes, you can capture only the final message printed by the `tf.classifier.train()` function by using TensorFlow's `tf.train.SessionRunHook` class to create a custom hook that prints only the final message.

```python
import tensorflow as tf

# Create a custom hook that prints only the final message
class FinalLossHook(tf.train.SessionRunHook):
    def end(self, session):
        # Get the final loss value from the graph
        final_loss_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('loss:0')
        final_loss = session.run(final_loss_tensor)
        # Print the final loss value
        print('Loss for final step:', final_loss)

# Create a classifier
classifier = tf.estimator.DNNClassifier(...)

# Train the classifier and capture only the final loss message
classifier.train(input_fn=train_input_fn, steps=num_steps, hooks=[FinalLossHook()])
```

<br>
In the code snippet above, we define a custom hook `FinalLossHook` that extends the `tf.train.SessionRunHook` class. The `end` method of the hook is called by TensorFlow after the final training step has completed, and it retrieves the final loss value from the graph using `tf.compat.v1.get_default_graph().get_tensor_by_name('loss:0')`. It then prints the final loss value.

We pass an instance of this custom hook to the `hooks` argument of the `classifier.train()` function, which causes the hook to be called after the final training step and print only the final loss message.

<span style="color:red;font-size:larger;font-weight:bold;">And of course:</span>

<span style="color:red;">AttributeError: module 'tensorflow._api.v2.train' has no attribute 'SessionRunHook'</span>

<br>

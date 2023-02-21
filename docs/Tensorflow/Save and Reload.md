## Save model

In the code example, we create a `model` and use it to make `predictions`.  is there a way to save the model to the file system, and read it back in, and make predictions?  and when we get the predictions, is there a typical way to display them?

Yes, you can save a trained model to the file system and load it back in to make predictions.

To save a model, you can use the `save()` method of the model object. This method saves the entire model to a file in a binary format that can be loaded back in later. Here's an example:

```py
from tensorflow import keras

# define and train a model
model = keras.Sequential([
  keras.layers.Dense(10, input_shape=(2,)),
  keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=10)

# save the model to a file
model.save('my_model.h5')
```

## Load Model

To load the saved model back in, you can use the `load_model()` function from the `keras.models` module. This function returns a new model object that has the same architecture and weights as the saved model. Here's an example:

```py
from tensorflow import keras

# load the saved model from a file
loaded_model = keras.models.load_model('my_model.h5')

# use the loaded model to make predictions
predictions = loaded_model.predict(x_test)
```

<br>
When you make predictions with a model, the output is usually a set of numbers. The exact format of these numbers depends on the problem you are trying to solve.

For example, if you are trying to **classify images of digits**, the output might be a **set of probabilities for each digit**.

e.g.

```
0.1  for 0
0.8  for 1
0.05 for 2
```

etc.


* Predict: price of house
* Output: a single number representing the predicted price

To **display the predictions** in a meaningful way, you might want to convert them into a more human-readable format.

For example, if you are **classifying** images of digits, you might want to **display the digit with the highest probability**.

If you are predicting the **price of a house**, you might want to display the **predicted price in dollars**.

The exact format of the display will depend on your specific problem and the preferences of your users.


## Troubleshooting

It complained about the **shapes not being equal**.  (Turns out we were using the wrong set of data.)

```py
# Show me the metadata
print("test length: {}, type: {}, shape: {}".format(len(y_test), type(y_test), np.shape(y_test)))

print("pred length: {}, type: {}, shape: {}".format(len(predictions), type(predictions), np.shape(predictions)))
```

<br>
Both test and train had a length of 10K.

What does this data look like?  Are there any weird values or NaN?

```py
import json
import numpy as np

# Do you really expect me to look through 10K values?

# Print to text file
with open('y_test.txt', 'w') as filehandle:
    json.dump(y_test.tolist(), filehandle)

with open('predictions.txt', 'w') as filehandle:
    json.dump(predictions.tolist(), filehandle)
```

<br>

OK, data looks like the shapes are different.

So it was suggested to **reshape** it (remember the data was wrong anyway, though):

```py
predicted_labels = predictions.argmax(axis=1)
```

<br>

Here's what it would look like to **graph** it (gigo):

```py
import matplotlib.pyplot as plt

# plot the true outputs vs. the predicted outputs
# plt.scatter(y_test, predictions) <- originally
plt.scatter(y_test, predicted_labels)
plt.xlabel("True outputs")
plt.ylabel("Predicted outputs")
plt.show()
```

<br>

**sklearn.metrics** `classification_report` worked (maybe it's not as strict):

```py
from sklearn.metrics import classification_report

# assuming y_test is the true labels and predicted_labels is the predicted labels
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predicted_labels))
```

<br>

Original suggested (messy) output:

```py
for i in range(len(predictions)):
    # print("Input: {}, True output: {}, Predicted output: {}".format(x_test[i], y_test[i], predictions[i]))
    print("Predicted output: {}".format(predictions[i]))
```

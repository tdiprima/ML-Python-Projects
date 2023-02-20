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

<br>
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

For example, if you are trying to classify images of digits, the output might be a set of probabilities for each digit (e.g. "0.1 for 0, 0.8 for 1, 0.05 for 2, etc.").

If you are trying to predict the price of a house, the output might be a single number representing the predicted price.

To display the predictions in a meaningful way, you might want to convert them into a more human-readable format.

For example, if you are classifying images of digits, you might want to display the digit with the highest probability.

If you are predicting the price of a house, you might want to display the predicted price in dollars.

The exact format of the display will depend on your specific problem and the preferences of your users.


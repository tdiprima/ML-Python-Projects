## Make nice display

When you make predictions with a model, the output is usually a set of numbers. The exact format of these numbers depends on the problem you are trying to solve.

For example, if you are trying to classify images of digits, the output might be a set of probabilities for each digit (e.g. "0.1 for 0, 0.8 for 1, 0.05 for 2, etc.").

If you are trying to predict the price of a house, the output might be a single number representing the predicted price.

To display the predictions in a meaningful way, you might want to convert them into a more human-readable format.

For example, if you are classifying images of digits, you might want to display the digit with the highest probability.

If you are predicting the price of a house, you might want to display the predicted price in dollars.

The exact format of the display will depend on your specific problem and the preferences of your users.

## ¿Problemas?

### Mismatched Array Lengths

When we plot the true outputs vs. the predicted outputs, it says: "x and y must be the same size".
How do I solve this?

The error message "x and y must be the same size" indicates that the lengths of the `y_test` and `predictions` arrays are not equal. <span style="color: red">Well, yeah!</span>

To fix this issue, you can make sure that both arrays have the same length.

Assuming that `y_test` and `predictions` are `NumPy` arrays, you can use the `len()` function to check their lengths:

```py
print(len(y_test))
print(len(predictions))
```

<br>
If the lengths are not equal, you need to make sure that the two arrays have the same shape.

One way to do this is to use NumPy's `reshape()` function to convert one of the arrays into the same shape as the other array.

For example, if `y_test` has shape `(n,)` and `predictions` has shape `(n, 1)`, you can reshape predictions to have shape `(n,)` as follows:

```py
predictions = predictions.reshape(-1)
```

<br>
This will flatten the `predictions` array and make it compatible with the `y_test` array.

Once you have made sure that both arrays have the same length, you can plot them as follows:

```py
plt.scatter(y_test, predictions)
plt.xlabel("True outputs")
plt.ylabel("Predicted outputs")
plt.show()
```

<br>
This should plot the true outputs vs. the predicted outputs without any errors.

### Both arrays are the same length.  Now what?

```
test length: 10000, type: <class 'numpy.ndarray'>
pred length: 10000, type: <class 'numpy.ndarray'>
```

<br>
If both `y_test` and `predictions` have the same length and contain 10,000 values, then it is likely that the error message is being caused by another issue in your code.

One possibility is that you have not imported the necessary libraries. Make sure that you have imported `matplotlib.pyplot` using the following command at the beginning of your code:

```py
import matplotlib.pyplot as plt
```

<br>
Another possibility is that you have not defined the variables `y_test` and `predictions` correctly. Make sure that they are NumPy arrays or Python lists that contain the actual output values and the predicted output values, respectively.

If you have already checked these possibilities and the error message persists, you can try printing the values of `y_test` and predictions to see if there are any unexpected values or NaNs. 

You can do this using the following commands:

```py
print(y_test)
print(predictions)
```

<br>
If you are still having trouble, please provide more information about the error message and your code, so that I can better understand the issue you are facing.


## y_test shape is (10000,) and predictions shape is (10000, 10)

I'm confused.  How do I fix that?

The shapes of `y_test` and `predictions` suggest that you are working with a multi-class classification problem where the target variable has 10 classes.

💡The shape of `(10000, 10)` for `predictions` means that it is a 2D array with 10 predicted probabilities for each of the 10,000 test examples.

To fix the issue of unequal shape, you need to convert the `predictions` array to a 1D array of predicted labels.

You can do this by finding the index of the maximum probability for each example using the `argmax` method in NumPy:

```py
predicted_labels = predictions.argmax(axis=1)
```

<br>
This will give you an array of predicted labels with shape (10000,).

Now, you can plot the true outputs vs. the predicted outputs as follows:

```py
plt.scatter(y_test, predicted_labels)
plt.xlabel("True outputs")
plt.ylabel("Predicted outputs")
plt.show()
```

<br>
This should plot the true outputs 🆚 the predicted outputs without any errors.

Note that the predicted labels will be integers between 0 and 9 (inclusive) corresponding to the 10 classes in your target variable.

### I see dots scattered everywhere in a box...

<span style="color:#000088; font-weight:bold">That worked!  I am now looking at a scatterplot.  How do I interpret this graph?  Like, specifically, if I see a dot at x=1, y=2, what does that tell me?</span>

Great to hear that the plot is working!

In a scatterplot of true vs. predicted values, each dot represents a single test example in your dataset.

The x-coordinate of the dot represents the true value of the target variable for that example.

The y-coordinate represents the predicted value.

If you see a dot at `x=1, y=2`, it means that for one of the test examples, the true value of the target variable was 1, while the predicted value was 2.

The distance between the dot and the line `y=x` (the diagonal line going from the bottom-left to the top-right corner of the plot) represents the error or the distance between the true value and the predicted value for that example.

A scatterplot can help you visualize the relationship between the true and predicted values, and assess the performance of your model.

Ideally, you want to see the dots clustered closely around the line y=x.

This indicates that the predicted values are close to the true values.

If the dots are widely scattered, it suggests that the model is not performing well and there is room for improvement.

You can also use metrics like mean squared error, mean absolute error, or **correlation coefficient** to quantify the performance of your model.

![](../images/Scott_Crawford.jpg)

### Is it possible that a scatterplot is not the best way to display results for the keras MNIST dataset?

*What is the MNIST dataset anyway?*

Yes, it is possible that a scatterplot is not the best way to display the results for the Keras MNIST dataset.

While a scatterplot can be useful for visualizing the performance of a regression model, the MNIST dataset is actually a classification problem, not a regression problem.

In a classification problem, the target variable is a categorical variable (e.g., in MNIST, the categories are the digits 0 to 9), and the goal is to predict the category of a new observation based on the values of the input features.

For a classification problem like MNIST, a confusion matrix or a classification report can be more informative for evaluating the model's performance.

A confusion matrix is a table that shows the number of correct and incorrect predictions for each class, while a classification report provides several evaluation metrics such as precision, recall, and F1-score for each class.

Regarding the MNIST dataset, it is a dataset of handwritten digits, commonly used for training and evaluating machine learning models.

The MNIST dataset contains a total of 70,000 images of handwritten digits, divided into 60,000 training images and 10,000 test images.

The images are grayscale, 28x28 pixels in size, and each image corresponds to a digit from 0 to 9.

The goal is to train a machine learning model that can correctly classify the digit shown in each image.

The MNIST dataset has become a standard benchmark for evaluating the performance of machine learning models, particularly for image classification tasks.

<br>


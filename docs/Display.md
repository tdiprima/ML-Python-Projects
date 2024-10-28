## Make a nice display

When you make predictions with a model, the output is usually a set of numbers. The exact format of these numbers depends on the problem you are trying to solve.

For example, if you are trying to classify images of digits, the output might be a set of probabilities for each digit (e.g. "0.1 for 0, 0.8 for 1, 0.05 for 2, etc.").

If you are trying to predict the price of a house, the output might be a single number representing the predicted price.

To display the predictions in a meaningful way, you might want to convert them into a more human-readable format.

For example, if you are classifying images of digits, you might want to display the digit with the highest probability.

If you are predicting the price of a house, you might want to display the predicted price in dollars.

The exact format of the display will depend on your specific problem and the preferences of your users.

![](../images/Scott_Crawford.jpg)

### a scatterplot is not the best way to display the results for the Keras MNIST dataset

While a scatterplot can be useful for visualizing the performance of a **regression** model, the MNIST dataset is actually a classification problem, not a regression problem.

In a **classification** problem, the target variable is a categorical variable (e.g., in MNIST, the categories are the digits 0 to 9), and the goal is to predict the category of a new observation based on the values of the input features.

For a classification problem like MNIST, a confusion matrix or a classification report can be more informative for evaluating the model's performance.

A **confusion matrix** is a table that shows the number of correct and incorrect predictions for each class, while a classification report provides several evaluation metrics such as precision, recall, and F1-score for each class.

*What is the MNIST dataset anyway?*

It is a dataset of handwritten digits, commonly used for training and evaluating machine learning models.

<br>


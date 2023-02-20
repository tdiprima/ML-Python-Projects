## Display predictions

OK, so how would we display `predictions` in our example?

In this example, predictions will be a numpy array of shape `(num_samples, 1)`, where `num_samples` is the number of samples in the test set. Each entry in the array is a predicted value for the corresponding input in the test set.

To display the predictions in a more meaningful way, you could iterate over the predictions array and print out each value. Here's an example:

```py
for i in range(len(predictions)):
  print("Input: {}, True output: {}, Predicted output: {}".format(x_test[i], y_test[i], predictions[i]))
```

<br>
This will print out each input in the test set, the true output, and the predicted output. You could modify this code to display the predictions in a different format that is more suitable for your specific problem.

### Is there any way to graph that?  It's printing a lot of numbers.

Yes, you can visualize the predictions using various graphing libraries in Python, such as **Matplotlib**, **Plotly**, or **Seaborn**.

Assuming you are doing a **regression** task (predicting a **continuous** value), a common way to visualize the predictions is to plot the **true outputs** versus the **predicted outputs**. 

You can use a **scatter plot** to do this, where the x-axis represents the true outputs and the y-axis represents the predicted outputs.

Each point in the plot represents a single data point in the test set.

Here's an example using Matplotlib:

```py
import matplotlib.pyplot as plt

# plot the true outputs vs. the predicted outputs
plt.scatter(y_test, predictions)
plt.xlabel("True outputs")
plt.ylabel("Predicted outputs")
plt.show()
```

<br>
This will display a scatter plot where each point represents a single data point in the test set.

The closer the points are to the diagonal line, the better the model's predictions are.

You can modify this code to change the appearance of the plot, such as adding a title or changing the color of the points.

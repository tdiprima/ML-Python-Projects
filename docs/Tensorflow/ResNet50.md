## What is ResNet50

<mark>**A type of deep neural network used for image 
recognition.**</mark>

*(What are other types of DNNs?)*

It has the ability to identify what objects are in an image, such as dogs, cats, or people.

The "50" in ResNet50 means that **the network has 50 layers.**

The more layers a network has, the more complex features it can learn, which can improve its ability to recognize images.

**How many layers** do they normally have?

Between 3 and 20 layers.

**Other DNNs:**

1. VGG16
1. Inception-v3
1. MobileNet
1. AlexNet
1. DenseNet
1. GoogLeNet
1. ResNet18
1. EfficientNet
1. YOLOv3
1. U-Net

### Residual connections <span style="font-size:27px;">🚇</span>

These connections allow the network to **"skip"** over some of the layers, which can help prevent the vanishing gradient problem. <span style="font-size:27px;">💨</span>

This problem can occur in very deep networks, where the **gradients** used to update the weights of the network become very small &ndash; **¡pequeño!** &ndash; making it difficult for the network to learn.

By using residual connections, ResNet50 can maintain a **strong gradient signal** 📶 throughout the network, making it easier to train and allowing it to learn more complex features.

This makes **ResNet50** a very powerful tool for **image recognition** tasks.


### Can I use bash "ack" to search for words ending with "50"?

Yes, you can use the ack command with a regular expression to search for words ending with "50". Here's an example command:

```sh
ack '\b\w+50\b'
```

This command uses the \b word boundary pattern to match whole words, and \w+ to match one or more word characters (letters, digits, or underscores). The + means "one or more", so \w+ matches any word character that appears one or more times. The \b pattern ensures that the word ends with "50" and is not followed by any other word character.

You can replace the ack command with grep or any other tool that supports regular expressions. Just make sure to enclose the regular expression in single quotes to prevent the shell from interpreting any special characters.


### Optimization algorithm called Adam 🍎

Adam, which is a popular choice for training neural networks.

Adam stands for **"Adaptive Moment Estimation"**, and it is a variant of stochastic gradient descent (SGD) that adapts the learning rate based on the first and second moments of the gradients.

Adam is known for its fast convergence and robustness to noisy gradients, and it is often used as the default optimizer in many machine learning frameworks, including TensorFlow and Keras.

**Optimization algorithms** used in machine learning:

* Gradient Descent
* Adagrad
* RMSprop
* Etc.


### Machine learning algorithms 🤖

* Linear regression
* Logistic regression
* Decision trees
* Random forests
* Support vector machines
* K-nearest neighbors
* Neural networks
* Etc.

It's worth noting that some machine learning algorithms use parameters that are set to a value of 50 or multiples of 50.

For example, the **k-nearest neighbors algorithm** uses a parameter **k** that specifies the number of nearest neighbors to consider when making predictions, and this value could be set to 50 or a multiple of 50.

However, the value of k is typically chosen based on cross-validation or other model selection techniques, rather than being set to a fixed value.


### R-squared value

The "R-squared" value is sometimes denoted as "R2".

In simple terms, the R-squared value is a statistical measure that represents the proportion of the variance in the output variable (i.e. the dependent variable) that is explained by the input variable(s) (i.e. the independent variable(s)).

The R-squared value ranges from 0 to 1, where a value of 0 means that none of the variance in the output variable is explained by the input variable, and a value of 1 means that all of the variance in the output variable is explained by the input variable(s).

### Example

For example, suppose you want to predict a person's height based on their weight.

The R-squared value of the prediction model would tell you how well the weight variable explains the variance in the height variable.

If the R-squared value is high, it means that weight is a good predictor of height, and there is a strong relationship between the two variables.

If the R-squared value is low, it means that weight is not a good predictor of height, and there is little to no relationship between the two variables.

In machine learning, the R-squared value is often used as a metric to evaluate how well a regression model fits the data.

A value close to 1 indicates a good fit, while a value close to 0 indicates a poor fit.

<br>

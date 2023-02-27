## Y = labels

<span style="color:maroon;font-size:larger;">y\_test and y\_train = True labels</span>

It is a **one-dimensional array** or vector of shape `(n_samples,)`, where `n_samples` is the number of data points in the training set.

<br>
<span style="color:maroon;font-size:larger;">So we got...</span>

* **supervised** machine learning

* **X:** input features (pictures)

* **y:** labels (0-9)

* **`y_train`:** train the model, known labels
* **`y_test`:** evaluate performance of model

In practice, the dataset is typically split randomly into training and testing sets, with a typical split being around **80% for training** and **20% for testing.**

### Example <span style="font-size:27px;">😸 🐶</span>

Let's say we have a classification problem where we want to predict whether an image contains a cat or a dog.

The true labels for the images would be either "cat" or "dog".

If we have a machine learning model that predicts the labels for these images, the predicted labels would be the model's predictions of "cat" or "dog" for each image.

In general, the true labels and predicted labels can be represented as arrays or tensors.

<span style="color:#0000dd;font-size:larger">Tensors</span>

For example, if we have a binary classification problem with **100 data points,** we might represent the true labels and predicted labels as NumPy arrays with **shape `(100,)`.**

## Target Variable <span style="font-size:35px;">🎯</span>

It's called the "target" variable because it's the variable we are **aiming to predict** or estimate.

**Target variable:** the thing that we want to predict.

* Target variable
* Output variable
* Response variable

### Example <span style="font-size:27px;">👵🏼👢</span>

Imagine you're trying to guess someone's **age** based on how **tall** they are. 

The **"target variable"** in this case would be the **person's age** because that's what you're trying to guess/predict.

The **height** of the person would be the **"input"** or **"predictor"** variable because it's the information you're using to make your guess.

### Example <span style="font-size:27px;">💵 🏠</span>

If we're trying to predict the **price of a house** based on its features, the target variable would be a **continuous variable** representing the price of the house.

<br>
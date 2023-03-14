## Train and Test Workflow

Our ideal model is using X\_test to predict y\_test.

We make predictions on our test dataset.

We learn (and optimize) on our training dataset.

So we pass X\_train to our model.

We learn patterns on the training data, to evaluate our model on the test data.

## Y = labels

<span style="color:maroon;font-size:larger;">y\_test and y\_train = True labels</span>

It is a **one-dimensional array** or vector of shape `(n_samples,)`, where `n_samples` is the number of data points in the training set.

<span style="color:#0000dd;font-size:larger">Tensors</span>

For example, if we have a binary classification problem with **100 data points,** we might represent the true labels and predicted labels as NumPy arrays with **shape `(100,)`.**

<span style="color:maroon;font-size:larger;">So we got...</span>

* **supervised** machine learning

* **X:** input features (pictures)

* **y:** labels (0-9)

* **`y_train`:** train the model, known labels
* **`y_test`:** evaluate performance of model

In practice, the dataset is typically split randomly into training and testing sets, with a typical split being around **80% for training** and **20% for testing.**


## Target Variable <span style="font-size:35px;">🎯</span>

It's called the "target" variable because it's the variable we are **aiming to predict** or estimate.

**Target variable:** the thing that we want to predict.

* Target variable
* Output variable
* Response variable

### Example <span style="font-size:27px;">💵 🏠</span>

If we're trying to predict the price of a house based on its features, the target variable would be a continuous variable representing the price of the house.

The square footage, number of baths, etc. would be the "input" or "predictor" variables.  Because it's the information you're using to make your guess.

<br>
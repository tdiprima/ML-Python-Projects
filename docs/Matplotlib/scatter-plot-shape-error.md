## Scenario:
I created a multi-label classification dataset using 

```py
# 4 classes
X, y = make_multilabel_classification(n_samples=100, n_features=2, n_classes=4)
```

If `pyplot.scatter()` expects a 1-dimensional array for the `c` argument, then why is it that

```py
plt.scatter(X[:, 0], X[:, 1], c=y)
```

actually works?

*I had no idea what the data looks like.  That's why I didn't know.*

## Answer:

The `plt.scatter(X[:, 0], X[:, 1], c=y)` function call works because the `c` argument expects a 1-dimensional array-like object with the **same length** as the **number of data points** in the scatter plot.

In this case, `y` is a 2-dimensional array with shape `(n_samples, n_classes)`, where each **row** corresponds to a **single data point**, and each **column** corresponds to a **binary label** indicating the presence or absence of a class.

<table border="1">
<tr><td width="25px;" align="center">1</td><td width="25px;" align="center">0</td><td width="25px;" align="center">1</td><td width="25px;" align="center">0</td></tr>
<tr><td width="25px;" align="center">0</td><td width="25px;" align="center">1</td><td width="25px;" align="center">1</td><td width="25px;" align="center">1</td></tr>
<tr><td width="25px;" align="center">1</td><td width="25px;" align="center">1</td><td width="25px;" align="center">0</td><td width="25px;" align="center">0</td></tr>
<tr><td colspan="4">Etc... 100 rows.</td></tr>
</table>

<br>

<span style="color:#0000dd;">I'm showing "binary", but most likely these values are float 32, I reckon.</span>

<span style="color:#0000dd;">So "data point" just means "sample."</span>

<span style="color:#0000dd;">And then the "labels" are obviously "classes".</span>

Now - When you pass `y` to the `c` argument of `plt.scatter()`, Matplotlib automatically converts the 2-dimensional `y` array into a 1-dimensional array by **flattening it row-wise**.

(i.e., concatenating the rows of `y` together into a single long array).

<span style="color:#0000dd;">So now you got a 1-D array with 400 elements in it.  Our table is now like a collapsable ruler that gets extended.</span>

This flattened array has length `n_samples * n_classes`, which is **the same as** the number of data points in the scatter plot.

<span style="color:#0000dd;">Ok, so let's not get confused by what Buddy is saying.  He's just saying there's 400 points in the scatter plot.  Cuz there's 100 samples with 4 classes.  "Data point" is `1 x 4`.</span>

<span style="color:#0000dd;">But according to his definition, then there should ALWAYS be the same number of points in the plot, "because" it's samples x classes.  Hellooo?</span>

So, even though `y` has a shape that does not match the expected shape for the `c` argument, Matplotlib is able to interpret the data correctly and generate a scatter plot with colors based on the binary labels in `y`.


## Rebuttal:

That doesn't make sense.  Then why `n_classes=5` does not work?  Wouldn't Matplotlib just flatten that too?  Is there a way to make sense of this mathematically, to make it clearer?  Like &ndash; we're plotting 100 samples, with 2 features, right? <span style="color:red;">Hecc no.</span>  So that's `100 * 2`.  Now `n_samples=4`, so 4 goes into 200 evenly.  But so does 5.  I'm confused.

## Answer:

You're correct that **Matplotlib will flatten any 2D array** that is passed to the `c` argument of `plt.scatter()`.

So, passing a `y` array with shape `(n_samples, n_classes)` should work regardless of the value of `n_classes`.

<span style="color:red;">Thank you!</span>

However, the resulting plot may not be visually informative or may not accurately represent the labels in `y`.

The issue with setting `n_classes=5` is that the labels in `y` will not be easily distinguishable using different colors in the scatter plot.

<span style="color:#0000dd;">(Cuz they all may be close shades of blue, for example?)</span>

With `n_classes=4`, Matplotlib **automatically selects a default <mark>color map</mark> with 4 distinct colors,** which can be used to differentiate between the different classes.

However, with `n_classes=5`, the default color map will only have 4 distinct colors, which means that one of the classes will be indistinguishable from another.

In general, it is recommended to use a [color map](https://matplotlib.org/stable/tutorials/colors/colormaps.html#sequential) that has enough distinct colors to **represent all of the classes** in the data.

One option is to manually specify a color map using the `cmap` argument of `plt.scatter()`.

For example, you can use the **viridis** color map, which has a large number of distinct colors, like this:

```py
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis').
```

In terms of the mathematics, you are correct that the size of the flattened `y` array will be equal to `n_samples * n_classes`.

In the case of `n_classes=4`, the flattened `y` array will have 400 elements, which is the same as the total number of data points in the scatter plot.

However, with `n_classes=5`, the flattened `y` array will have 500 elements, which is **not divisible by 3** (the number of color components used by Matplotlib to represent each data point in the scatter plot).

<span style="color:#0000dd;">Alright, so you EITHER have to have the "same number", which still makes no sense, OR it has to be divisible by 3; assuming "components" means "color channels."</span>

This means that Matplotlib will not be able to **assign a unique color to each label in `y`**, which can result in the plot being difficult to interpret.

<h1 align="center">Archive</h1>

## I tried using it in multilabel\_class.py

I made a multi-label classification using

```py
# 5 classes
X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=5)
```

Then I tried to plot it using

```py
plt.scatter(X[:, 0], X[:, 1], c=y)
```

but I got an `invalid_shape_exception` that says "ValueError: 'c' argument has 500 elements, which is inconsistent with 'x' and 'y' with size 100."  How do I fix this?

The `plt.scatter` function expects `c` to be an **array of color values**, with **one value for each point** in the plot.

In your case, `y` is a multi-label classification target with 5 classes, so it has shape `(100, 5)`.

To plot the data with different colors for each class, you can use the <mark>**`np.argmax`**</mark> function to **get the index of the maximum value** for each row of y.

This will give you an **array of integer labels** that you can use as the `c` argument to `plt.scatter`.

Here's how you can modify your code to plot the data:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification

# create the data
X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=5)
print("y.shape:", y.shape)  # (100, 5)

# get the integer labels
labels = np.argmax(y, axis=1)  # <- numpy
print("labels.shape:", labels.shape)  # (100,)

# plot the data
plt.scatter(X[:, 0], X[:, 1], c=labels)  # <- See?
# plt.scatter(X[:, 0], X[:, 1], c=y)

# show the plot
plt.show()
```

## y shape confusion, 3 classes

I created a multi-label classification dataset using

```py
# 3 classes
X, y = make_multilabel_classification(n_samples=100, n_features=2, n_classes=3)
```

`y` has a shape of `(100, 3)`.

Then I created a scatter plot using

```py
plt.scatter(X[:, 0], X[:, 1], c=y)
```

It works.

On the other hand, I then created a multi-label classification dataset using

```py
# 5 classes
X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=5)
```

The shape of `y` is `(100, 5)`.  I created a scatter plot the same way.  My question is, why does `y` with shape `(100, 3)` work, but `y` with shape `(100, 5)`, doesn't work?


The reason why `plt.scatter(X[:, 0], X[:, 1], c=y)` works with `y` of shape `(100, 3)` but not with `y` of shape `(100, 5)` is because `plt.scatter()` **expects a 1-dimensional array** for the `c` argument, which is used to specify the color of each point in the scatter plot.

When `y` has a shape of `(100, 3)`, it means that each sample in `X` has three labels.

In this case, `plt.scatter()` interprets the `c` argument as a sequence of three colors, one for each label, and assigns a color to each point based on the label it belongs to.

However, when `y` has a shape of `(100, 5)`, it means that each sample in `X` has five labels.

In this case, `plt.scatter()` cannot interpret the `c` argument as a sequence of five colors, and raises a ValueError indicating that the shape of `c` does not match the shape of `X`.

To plot a scatter plot with `y` of shape `(100, 5)`, you can use a different visualization technique, such as plotting each label separately, or using a different color scheme that can represent multiple labels.


## It sounds like you're saying...

It sounds like you're saying that there's a limit to how pyplot interprets the `c` argument.  In other words, c with shape (100, 1) is ok, and c with shape (100, 3) is ok.  But anything more than (100, 3) is not ok?

Yes, that's correct. When you pass an array to the `c` argument in `plt.scatter()`, it expects the array to be one-dimensional **OR** have the same length as the number of points in the plot.

When you pass an array with more than one dimension, such as `(100, 3)` or `(100, 5)`, `plt.scatter()` interprets it as a **color sequence** for each point, where each row represents a color for a point.

So, for example, a `(100, 1)` array will be interpreted as a sequence of colors, one for each point, and a `(100, 3)` array will be interpreted as a sequence of 3 colors, one for each label, for each point.

However, when you pass an array with more than 1 dimension **AND** the number of rows is not equal to the number of points in the plot, such as a `(100, 5)` array, `plt.scatter()` will raise a `ValueError` because it cannot interpret the array as a color sequence for each point.


## n_classes=4 works 🤷‍♂️

Playing around with the number of classes.

```python
# N classes
X, y = make_multilabel_classification(n_samples=100, n_features=2, n_classes=N)
```

I tried n_classes=1-8, here's what works:

* 4
* 3
* 1


## Grr! 🐯

I thought I understood it, but I don't.  How do I know which numbers will work for pyplot, when using

```py
make_multilabel_classification(n_samples=N1, n_features=N2, n_classes=N3)
```

Please explain.

I apologize if my previous explanations were not clear enough. I'll try to explain in simpler terms.

When you use `make_multilabel_classification()` to generate a multilabel classification dataset, you specify the number of **samples**, number of **features**, and number of **classes**.

The output labels, `y`, will have a shape of `(N1, N3)`, where N1 is the number of samples and N3 is the number of classes.

<span style="color:#0000dd;font-size:larger;">Using label and class interchangeably.</span>

**Each label** is represented by a **binary value**, which indicates whether a given sample belongs to that class or not.

When you try to plot the data using `plt.scatter()`, you can use the `c` argument to specify the color of each point in the scatter plot, based on its **corresponding label value.**

However, `plt.scatter()` expects a one-dimensional array for the `c` argument, which means that the **length** of the array should be **equal** to the **number of points** in the scatter plot.

If the shape of `y` is `(N1, 1)`, where each **label** has only **one binary value**, you can pass y directly to the `c` argument of `plt.scatter()`.

In this case, `y` is interpreted as a sequence of colors, one for each point in the scatter plot.

If the shape of `y` is `(N1, N2)`, where each label has more than one binary value, you need to **convert y to a one-dimensional array** before passing it to the `c` argument of `plt.scatter()`.

You can use the `np.argmax()` function to find the index of the label with the highest value for each sample,

<span style="color:green;font-size:larger;">Crickets... 🦗 And I still don't have what I'm lookin' fer. 🌙 🐺</span>

<br>

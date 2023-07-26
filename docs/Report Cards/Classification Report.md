## Classification Report

You can use scikit-learn's `classification_report` function to generate a comprehensive classification report **for your model's predictions.**

To use this function, you'll need the true labels (i.e., `y_test`) and the `predicted` labels.

### Code

Here's an example code snippet that shows how to generate a classification report using `scikit-learn`.

```py
from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_labels))
```

<br>
This will output a report containing evaluation metrics for each class (i.e., digit) in the MNIST dataset.

The report includes precision, recall, F1-score, and support for each class.

The **support** is simply the number of observations in each class.

The **precision** measures how often the model correctly predicted the positive class (i.e., the digit).

The **recall** measures how well the model identified all positive instances.

The **F1-score** is the harmonic mean of precision and recall, and it provides a balanced measure of the model's performance.

### Report output

`print(classification_report(y_test, y_pred))`

```
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.97      0.97      1032
           3       0.98      0.98      0.98      1010
           4       0.97      0.98      0.98       982
           5       0.98      0.98      0.98       892
           6       0.97      0.99      0.98       958
           7       0.97      0.98      0.98      1028
           8       0.97      0.97      0.97       974
           9       0.98      0.96      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
```

## Seeker's Precision & Recall

Alright, let's take a step back and imagine you're playing a game of hide and seek. Your job as the seeker is to find as many of your friends as possible who are hiding. 

1. **Precision**: Let's say you think you found a hiding spot. You shout, "Gotcha!" But alas, it's just a bush. If you keep shouting "Gotcha!" at every bush and rock, you might eventually find your friends, but you also "found" a lot of things that aren't your friends. Precision is **how often you're correct** when you shout "Gotcha!" In other words, it's the ratio of true positives (when you correctly identify a friend) to all identified positives (the number of times you shouted "Gotcha!").

2. **Recall**: Imagine you finished the game and went back home, only to realize you left a couple of your friends still hiding in the park. Oh no! Recall, also known as **sensitivity**, is the ability of the seeker (you) to find all the hiders (your friends). It's the ratio of true positives (friends you correctly found) to all actual positives (all friends who were hiding). 

In machine learning, when we make a model to classify things (like deciding if an email is spam or not), precision and recall are important measurements. 

- High precision means that when our model says something is true (like an email is spam), it is very likely to be correct.

- High recall means that our model is good at catching all the true cases (like it can identify most of the actual spam emails).

So just like in the hide and seek game, you want to be as sure as possible when you find a hiding spot (high precision) and you also want to find all of your friends (high recall). But remember, sometimes when you try to find all your friends (high recall), you might shout "Gotcha!" at a lot of bushes (low precision). This is called the precision-recall trade-off.

<br>

<!--<span style="color:#000088; font-weight: bold; font-size: 1.2em;">-->
# Output Variables

### Categorical (or "discrete"), and continuous.

## Categorical

<span style="color:#000088;font-weight:bold;">Discrete / categorical</span> = things that can be counted or labeled.

A discrete or categorical output variable is a type of variable that can only take on a limited number of values or categories.&nbsp;&nbsp;In machine learning, this type of output variable is typically associated with classification tasks, where the goal is to predict the category or class to which an input belongs.

For example, consider the task of predicting whether an image contains a dog or a cat.&nbsp;&nbsp;In this case, the output variable is categorical, with only two possible categories: "dog" or "cat".&nbsp;&nbsp;Another example might be predicting whether a customer will churn or not from a telecom company.&nbsp;&nbsp;In this case, the output variable is again categorical, with only two possible categories: "churn" or "not churn".

Other examples of categorical output variables could include predicting the type of flower in a photograph, classifying the sentiment of a text message as positive or negative, or predicting the outcome of a sports game as win, lose, or draw.

## Continuous

<span style="color:#000088;font-weight:bold;">Continuous</span> can take on any value within a certain range.

In contrast, a continuous output variable can take on any numeric value within a certain range, such as predicting the price of a house or the temperature outside.&nbsp;&nbsp;It's important to identify the type of output variable you're working with, as this can help you determine which machine learning algorithm to use and how to evaluate the performance of your model.

## Code

```py
# discrete variable (counting apples)
num_apples = 5
print("There are", num_apples, "apples in the basket")

# categorical variable (color of a car)
car_color = "red"
print("The car is", car_color)

# continuous variable (weight of an apple)
apple_weight = 0.153
print("The weight of the apple is", apple_weight, "kg")
```

<br>

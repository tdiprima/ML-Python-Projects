## "gradient descent", weights, and biases

Sure! Let me break it down for you.

Imagine you're playing a game of **"guess my number"** with a friend. Your friend thinks of a number between 1 and 100, and you have to guess it. After each guess, your friend tells you whether your guess was too high or too low, and you adjust your next guess accordingly. This is kind of like what a computer does when it's trying to learn something.

### Weights & Biases ⚖️

In machine learning, we use something called "weights" and "biases" to **make guesses** and improve our accuracy. 

**Weights** are kind of like the knobs on a radio - they control how much each piece of information affects the final result.

**Biases** are like the starting point for the guesses - they give the model an initial idea of what the answer might be.

### Gradient Descent 🏂

Now, let's talk about "gradient descent." Gradient descent is a fancy way of saying that the computer is trying to **figure out the best possible values for the weights and biases** to make the most accurate guesses.

It does this by starting with some initial guesses for the weights and biases, making a guess, checking how close it was to the actual answer, and then adjusting the weights and biases based on how far off it was.

The **"gradient"** part of gradient descent comes from the fact that the computer is looking at the slope, or gradient, of a curve that represents how far off its guesses are from the actual answer.

By **following the slope downhill**, the computer can find the weights and biases that give the most accurate guesses.

So, in summary, weights and biases are like knobs and starting points that a computer uses to make guesses, and gradient descent is the process the computer uses to figure out the best possible values for those knobs and starting points.

### so "weights" are kind of like "this feature doesn't count as much"? 🤔

Yes, that's a good way to think about it!

Weights allow the machine learning model to assign different **levels of importance** to different **features** of the input data when making a prediction.

For example, if you're trying to predict the price of a house, the **number of bedrooms** might be more important than the **color of the front door**. 🛌 🆚 🚪

So, the model can assign a **higher weight** to the number of bedrooms to reflect its **importance** in making the prediction.

Weights can actually take on **any real number**, positive or negative, so they don't have to be between 0 and 1.

The **values** of the weights are learned by the machine learning algorithm during training, and the optimal values depend on the specific problem and the characteristics of the data being used. 

During training, the algorithm **adjusts** the weights to **minimize the error** between the predicted output and the true output. Once training is complete, the weights are fixed and used to make predictions on new, unseen data.

### oh.  so i don't have to pick the number, the algorithm decides that?

That's right!

In machine learning, we let the algorithm figure out the best values for the weights and biases automatically.

This is one of the advantages of using machine learning - we don't have to manually choose the weights and biases, which can be time-consuming and error-prone. Instead, we give the algorithm some example data and the correct answers, and it learns how to make accurate predictions by adjusting the weights and biases.

This is typically done through a process called **"training,"** where the algorithm iteratively adjusts the weights and biases based on the error between its predictions and the correct answers, until the error is minimized. Once the training is complete, the algorithm has learned the optimal values of the weights and biases, and can use them to make predictions on new, unseen data.

<scpan style="color:blue">Note: This is because Andrew Ng had us code sh!t from scratch, first, but real-world application is different.  So I got confused.<span>

### ok. so weights and biases are related to features?

Yes, that's right! Both weights and biases are used to make predictions in machine learning models and are related to the features of the input data.

<mark>**Weights**</mark> are used to determine the **importance** of each feature in making a prediction.

💡The model **multiplies each input feature** by its corresponding weight and sums the products to make a prediction.

The weights are learned during training to minimize the error between the predicted output and the true output.

<mark>**Biases**</mark>, on the other hand, are used to **shift the predicted values up or down.**

They act as a kind of **offset** that helps to account for factors that aren't captured by the input features.

For example, in the case of predicting house prices, there might be a baseline price that **doesn't depend on** any of the features, such as the cost of construction or land value in the area.

The bias term helps to capture this baseline price. Like the weights, the bias term is also learned during training to minimize the error between the predicted output and the true output.


### so bias is kind of like a handicap in golf? ⛳️

Yes, that's a good analogy!

In golf, a handicap is a system used to **level the playing field** between players of different skill levels. Similarly, biases in machine learning models are used to account for factors that aren't captured by the input features and can help to level the playing field **between different input samples**.

Just as a golfer's handicap is subtracted from their final score to account for their skill level, **the bias term is added to the predicted output of the model** to account for factors that aren't captured by the input features.

The bias term helps to make the model more **flexible** and able to fit a wider range of input data. Like the weights, the bias term is learned during training and adjusted to minimize the error between the predicted output and the true output.

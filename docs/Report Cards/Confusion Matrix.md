### Confusion Matrix

A confusion matrix is a way to see **how well a machine learning model** is doing at **classifying things correctly.**

Imagine you have a bunch of **pictures of animals** and you want to teach a computer program to **recognize which animal** is in each picture.

Let's say you have pictures of dogs, cats, and birds.

Now, let's say you've trained your computer program on a bunch of pictures and it's time to see how well it does at **recognizing new pictures it hasn't seen before.** You can use a confusion matrix to help you see how well it did.

The **confusion matrix is like a table** that shows you how many pictures the computer program got right and how many it got wrong.

Along the **top** of the table, you **list the actual animals** in the pictures (dogs, cats, and birds).

Along the **side** of the table, you list **what the computer program guessed** the animal was (also dogs, cats, and birds).

Now you can fill in the table with **how many pictures** the computer program got right and how many it got wrong.

For example, if the computer program guessed a picture was a dog and it was actually a cat, you would put that in the table.

Using the confusion matrix, you can see **how many times** the computer program got each animal right and wrong.

You can also calculate some **statistics** like **accuracy** (how often it got the animal right) and **precision** (how often it got the animal right when it said it was that animal).

Overall, the confusion matrix helps you see how well your machine learning model is doing and where it might need some improvement.

## Example

Here's a make-believe confusion matrix table based on the above  example:

<table><thead><tr><th></th><th>Actual: Dog</th><th>Actual: Cat</th><th>Actual: Bird</th></tr></thead><tbody><tr><td>Guessed: Dog</td><td align="right">10</td><td align="right">2</td><td align="right">1</td></tr><tr><td>Guessed: Cat</td><td align="right">3</td><td align="right">8</td><td align="right">2</td></tr><tr><td>Guessed: Bird</td><td align="right">0</td><td align="right">1</td><td align="right">9</td></tr></tbody></table>


### Dog guesses

The computer program guessed that:

* 13 pictures were dogs, and out of those,

* 10 were actually dogs, 

* 2 were cats, and 

* 1 was a bird.

### Cat guesses

* 13 pictures were cats, and out of those 
* 8 were actually cats, 
* 3 were dogs, and 
* 2 were birds.

### Bird guesses

* 10 pictures were birds, and out of those 
* 9 were actually birds, and 
* 1 was a cat.

From this confusion matrix, you can see that the **computer program was best at recognizing birds** (it got 9 out of 10 correct).

But it had a harder time **distinguishing between dogs and cats** (it got 10 out of 13 dogs correct, and 8 out of 13 cats correct).

By looking at these results, you can think about how to improve the computer program's performance, such as:

* adding more training data, or 
* tweaking the algorithm.

## tf 

In TensorFlow, you can use the `confusion_matrix` function to generate a confusion **matrix**, and then use the `f1_score` function to calculate the **F1-score for each class.**

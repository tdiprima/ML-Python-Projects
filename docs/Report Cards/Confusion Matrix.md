## Confusion Matrix

A confusion matrix is a way to see **how well a machine learning model** is doing at **classifying things correctly.**

### Example 🐱 🐶 🦆

Ok, so we're playing "recognize the animal."

Let's say you've trained your computer program on a bunch of pictures and it's time to see how well it does at recognizing new pictures it hasn't seen before.

You can use a confusion matrix to help you see how well it did.

The **confusion matrix is like a table** that shows you how many pictures the computer program got right and how many it got wrong.

**Top:** actual animals

**Side:** guessed animals

Using the confusion matrix, you can see **how many times** the computer program got each animal right and wrong.

You can also calculate some **statistics** like **accuracy** (how often it got the animal right) and **precision** (how often it got the animal right when it said it was that animal).

## Table

Here's a make-believe confusion matrix table based on the above  example:

<table><thead><tr><th></th><th>Actual: Dog</th><th>Actual: Cat</th><th>Actual: Bird</th></tr></thead><tbody><tr><td>Guessed: Dog</td><td align="right">10</td><td align="right">2</td><td align="right">1</td></tr><tr><td>Guessed: Cat</td><td align="right">3</td><td align="right">8</td><td align="right">2</td></tr><tr><td>Guessed: Bird</td><td align="right">0</td><td align="right">1</td><td align="right">9</td></tr></tbody></table>

<br>

The computer program guessed that...

<span style="font-size:20px;">Dogs</span> <span style="font-size:27px;">🐶</span>

* 13 pictures were dogs, and out of those,
* 10 were actually dogs, 
* 2 were cats, and 
* 1 was a bird.

<span style="font-size:20px;">Cats</span> <span style="font-size:27px;">🐱</span>

* 13 pictures were cats, and out of those 
* 8 were actually cats, 
* 3 were dogs, and 
* 2 were birds.

<span style="font-size:20px;">Birds</span> <span style="font-size:27px;">🦆</span>

* 10 pictures were birds, and out of those 
* 9 were actually birds, and 
* 1 was a cat.

From this confusion matrix, you can see that <mark>the **computer program was best at recognizing birds** (it got **9 out of 10** correct).</mark>

It got 10 out of 13 dogs correct.

It got 8 out of 13 cats correct.

By looking at these results, you can think about how to **improve** the computer program's performance, such as:

* Adding more training data
* or
* Tweaking the algorithm

<br>

## cross-entropy

In machine learning, we often want to teach a computer program how to **classify** things correctly.

For example, if we want the program to be able to tell the difference between pictures of **cats** and pictures of **dogs**, we need to train it using a lot of examples of each.

When we're training the program, we want to make sure that it's **getting better** over time.

One way we can **measure how well** it's doing is by using something called cross-entropy.

**Cross-entropy** is like a **score** that tells us **how close** the program's guesses are to the correct answers.

The **closer** the guesses are, the **lower** the cross-entropy **score**.

Imagine you have a friend who's trying to **guess what animal** you're thinking of, and you can only give them "yes" or "no" answers to their questions.

If they ask you "does it have fur?", and the animal you're thinking of does have fur, you would say "yes". 🐻

If they guess "snake", you would say "no". 🐍

Cross-entropy is like **counting how many "no" answers** your friend gets before they finally guess the right animal.

The **lower the number of "no" answers**, the **closer** your friend was to guessing the right animal.

So in machine learning, we use cross-entropy to see how close the program's guesses are to the correct answers.

The lower the cross-entropy score, the better the program is doing!

## Why is it called "Cross entropy"?

Cross entropy is a fancy term that comes from math and science, and it's used to <mark>**measure the difference between two sets of probabilities.**</mark>

### Entropy 🤨

**Entropy** is a **measure** of how much **uncertainty** or randomness there is in a set of data.

### Cross 🎿

Now, imagine you have two sets of data, and you want to **compare how similar** they are to each other.

You can use cross entropy to do this.

The **"cross"** part of the name comes from the fact that you're **comparing two different sets of data**, or **crossing them.**

So, cross entropy is a way of measuring how different two sets of probabilities are from each other.

It's called "cross" because you're comparing two different sets, and "entropy" because you're measuring how much uncertainty or randomness there is in the data.

## Tool Time! 🧰

Here's a simple Python example of how to calculate cross-entropy between two probability distributions:

```py
import numpy as np

# Define two probability distributions
p = np.array([0.2, 0.3, 0.5]) # true distribution
q = np.array([0.3, 0.3, 0.4]) # predicted distribution

# Calculate cross-entropy
cross_entropy = -np.sum(p * np.log(q))

print(f"Cross-entropy: {cross_entropy:.4f}")

```

In this example, we first define two probability distributions, `p` and `q`.

`p` represents the **true** distribution.

`q` represents a **predicted** distribution. 

We then use the formula for cross-entropy:

```
H(p, q) = -∑ p(x) log(q(x))
```

to calculate the cross-entropy between `p` and `q`.

Finally, we print the resulting cross-entropy value.

Note that in this example, we use **NumPy** arrays to represent the probability distributions and to perform the **element-wise multiplication** and **summation** required by the cross-entropy formula.

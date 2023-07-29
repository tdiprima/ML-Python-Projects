## Cross-Entropy <span style="font-size:35px;">🎿</span>

When we're training the program, we want to make sure that it's getting better over time.

We use cross-entropy to see how close the program's guesses are to the correct answers.

The lower the cross-entropy score, the better the program is doing!

It's like counting how many "no" answers the machine gets before it finally guesses the right thing.

**Cross-entropy** is a measure of the **difference** between two probability distributions, typically used in machine learning for classification tasks. It is calculated as the negative sum of the true label multiplied by the log of the predicted label. Cross-entropy measures the dissimilarity between the predicted probability distribution and the true probability distribution.

<span style="color:#880000;font-size:larger;">Cross-entropy = difference; probability distributions.</span>

<span style="color:#880000;font-size:larger;">Dice = similarity; segmentations.</span>


## Why is it called "Cross entropy"?

Well, it's used to measure the difference between two sets of probabilities.

**Entropy** = how much uncertainty or randomness

Imagine you have two sets of data, and you want to compare how similar (different?) they are to each other.

The "cross" part of the name comes from the fact that you're comparing two different sets of data, or **crossing them.** <span style="font-size:27px;">🎿</span>

## Code <span style="font-size:27px;">👩🏻‍💻</span>

**`p`** = true distribution

**`q`** = predicted distribution

Calculate cross-entropy between two probability distributions:

```py
import numpy as np

# Define two probability distributions
p = np.array([0.2, 0.3, 0.5]) # true distribution
q = np.array([0.3, 0.3, 0.4]) # predicted distribution

# Calculate cross-entropy
cross_entropy = -np.sum(p * np.log(q))

# Step through it:
a = np.log(q)
print("\npreds:", q)
print("log of preds:", a)

b = p * a
print("\ntrue preds:", p)
print("true preds x log:", b)

c = -np.sum(b)
print("\nResult:", c)  # 1.0601317681000455

print(f"\nCross-entropy: {cross_entropy:.4f}")  # 1.0601
```

## Formula

Calculate the cross-entropy between `p` and `q`.

```
H(p, q) = -∑ p(x) log(q(x))
```

## Element-wise

We use NumPy arrays to perform the element-wise multiplication and summation required by the cross-entropy formula.

<span style="color:#0000dd;">See:</span> [log.md](../Logarithm/log.md)

<br>

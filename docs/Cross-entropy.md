## Cross-Entropy

When we're training the program, we want to make sure that it's getting better over time.

We use cross-entropy to see how close the program's guesses are to the correct answers.

The lower the cross-entropy score, the better the program is doing!

It's like counting how many "no" answers the machine gets before it finally guesses the right thing.

### Why is it called "Cross entropy"?

Well, it's used to measure the difference between **two sets** of probabilities.

**Entropy** = how much uncertainty or randomness

Imagine you have two sets of data, and you want to compare how similar (different?) they are to each other.

The "cross" part of the name comes from the fact that you're comparing two different sets of data, or **crossing them.** <span style="font-size:30px;">🎿</span>

### Code <span style="font-size:27px;">👩🏻‍💻</span>

How to calculate cross-entropy between two probability distributions:

```py
import numpy as np

# Define two probability distributions
p = np.array([0.2, 0.3, 0.5]) # true distribution
q = np.array([0.3, 0.3, 0.4]) # predicted distribution

# Calculate cross-entropy
cross_entropy = -np.sum(p * np.log(q))

print(f"Cross-entropy: {cross_entropy:.4f}")

```

<br>
`p` = true distribution

`q` = predicted distribution

Calculate the cross-entropy between `p` and `q`, using the **formula for cross-entropy:**

```
H(p, q) = -∑ p(x) log(q(x))
```

<br>
We use NumPy arrays to perform the **element-wise multiplication** and **summation** required by the cross-entropy formula.

<span style="color:#0000dd;font-size:larger;">See:</span> [log.md](Math/log.md)

<br>

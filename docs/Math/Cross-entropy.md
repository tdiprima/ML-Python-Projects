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

print(f"Cross-entropy: {cross_entropy:.4f}")  # 1.0601
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

### Cross-entropy vs Dice

Aren't they kinda the same thing?

Well, they both work on sets, and I thought that they both measure dissimilarity, but they don't.

**Cross-entropy** is a measure of the difference between two probability distributions, typically used in machine learning for classification tasks. It is calculated as the negative sum of the true label multiplied by the log of the predicted label. Cross-entropy measures the **dissimilarity** between the predicted probability distribution and the true probability distribution.

**Dice coefficient,** on the other hand, is a **similarity** measure between two sets that measures the overlap between them. It is often used in image segmentation and medical imaging applications. The Dice coefficient is calculated as twice the intersection of the two sets divided by the sum of their sizes.

While both cross-entropy and Dice coefficient are used in machine learning, they are used for different purposes and measure different aspects of the models' performance. Cross-entropy measures the dissimilarity between predicted and true probability distributions, while Dice coefficient measures the overlap between two sets.

<br>

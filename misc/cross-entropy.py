"""
Calculates the cross-entropy between two probability distributions and prints the result.
"""
import numpy as np


def process(p1, q1):
    # Calculate cross-entropy
    cross_entropy = -np.sum(p1 * np.log(q1))

    print(f"Cross-entropy: {cross_entropy:.4f}")


# Define two probability distributions
p = np.array([0.2, 0.3, 0.5])  # true distribution
q = np.array([0.3, 0.3, 0.4])  # predicted distribution
process(p, q)

p = np.array([0.9, 0.9, 0.9])
q = np.array([0.3, 0.3, 0.4])
process(p, q)

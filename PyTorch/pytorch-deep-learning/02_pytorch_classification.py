import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles

# Make 1000 samples
n_samples = 1000

"""
Make-circles dataset
Make a large circle containing a smaller circle in 2d.
A simple toy dataset to visualize clustering and classification algorithms.
"""
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print("\nDS len:", len(X), len(y))

print(f"\nFirst 5 samples of X:\n {X[:5]}")
print(f"\nFirst 5 samples of y:\n {y[:5]}")

# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

print("\nCircles head", circles.head(10))

print("\nLabel count:", circles.label.value_counts())

print("\nX y shape:", X.shape, y.shape)

# Visualize, visualize, visualize
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.show()

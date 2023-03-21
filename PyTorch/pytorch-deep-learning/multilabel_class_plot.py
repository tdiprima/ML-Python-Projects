import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification


# Define a function to plot the dataset
def plot_dataset(X, y):
    plt.title("Make Multilabel Classification")
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()


# Generate a random multilabel classification dataset with 100 samples and 2 features
X, y = make_multilabel_classification(n_samples=100, n_features=2, n_classes=3)
print("y.shape:", y.shape)  # (100, 3)

# Plot the dataset
plot_dataset(X, y)
plt.show()

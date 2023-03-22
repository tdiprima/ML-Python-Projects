import matplotlib.pyplot as plt


# VISUALIZE
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Training data = blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Test data = green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Predictions = red
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Legend
    plt.legend(prop={"size": 14})

    plt.show()

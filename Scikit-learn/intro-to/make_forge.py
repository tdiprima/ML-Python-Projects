# -*- coding: utf-8 -*-
"""
https://www.tarleton.edu/faculty/agapie/documents/cosc4360/make_forge.txt
https://github.com/amueller/mglearn/blob/master/mglearn/datasets.py#L12
"""

import numpy as np
from sklearn.datasets import make_blobs


def make_forge():
    # a carefully hand-designed dataset lol
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    y[np.array([7, 27])] = 0
    # mask = np.ones(len(X), dtype=np.bool)  # deprecated
    mask = np.ones(len(X), dtype=bool)
    mask[np.array([0, 1, 5, 26])] = 0
    X, y = X[mask], y[mask]
    return X, y


# Example of use:
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from matplotlib import pyplot as plt

    X, y = make_forge()

    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)

    print(f"\npredict: {clf.predict(np.expand_dims(X[0], axis=0))}")
    print(f"\npredict_proba: {clf.predict_proba(np.expand_dims(X[0], axis=0))}")

    # Wistia, RdBu, RdBu_r
    plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap="Wistia")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()

    print(f"\nX.shape: {X.shape}")

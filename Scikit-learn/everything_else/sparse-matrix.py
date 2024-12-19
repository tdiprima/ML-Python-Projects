"""
Generates a random data matrix, enforces a threshold value, creates corresponding labels, counts unique labels and then
flips the labels, aiming to perform training on a data set and test the accuracy of a model. However, with the small
size of this data set, the model performance may not be satisfactory.
"""
import numpy as np

# Create 10 samples with 5 feature-sets each
X = np.random.random((10, 5))  # random.sample()?

# Shape of X
print(f"X.shape: {X.shape}")

# Set all the elements in the array X that are less than 0.7 to 0
X[X < 0.7] = 0

# Create corresponding labels(M, F)
arr = ['M', 'F', 'M', 'M', 'M', 'F', 'M', 'F', 'M', 'F']
y = np.array(arr)

counts = np.unique(arr, return_counts=True)
print(f"arr: {arr}")

# Modify array in-place.
for i in range(len(arr)):
    if arr[i] == 'M':
        arr[i] = 'F'
    elif arr[i] == 'F':
        arr[i] = 'M'

print(f"arr: {arr}")

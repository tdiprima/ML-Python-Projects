"""
Traditional example dog vs cat classification
3 images of dogs and 3 images of cats; images have only 2 pixel values x, y (so we can easily plot them)
Import -> Initialize -> Fit(Train) -> Predict
https://medium.com/@parthvadhadiya424/hello-world-program-with-scikit-learn-a869beb55deb
"""
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors  # import model building

simplefilter(action='ignore', category=FutureWarning)

# Python dict that contains two classes(dog, cat) with three samples each
# and two features (maybe height, width)
training_set = {'Dog': [[1, 2], [2, 3], [3, 1]], 'Cat': [[11, 20], [14, 15], [12, 15]]}
print(f"\nTraining set type: {type(training_set)}, len: {len(training_set)}")
print(f"Keys: {training_set.keys()}")
print(f"\nDog: {type(training_set['Dog'])}, len: {len(training_set['Dog'])}")
print(f"Shape: {np.array(training_set['Dog']).shape}")  # todo: First number is # samples, 2nd number is # features.

# Testing set that we will use for prediction
testing_set = [15, 20]
print(f"\nTest set type: {type(testing_set)}, len: {len(testing_set)}")
print(f"Shape: {np.array(testing_set).shape}")

# PLOT TRAINING SET
c = 'x'
for data in training_set:
    print(f"\nTrain data:", data)
    print(training_set[data])
    for i in training_set[data]:
        plt.plot(i[0], i[1], c, color='c')
    c = 'o'
plt.show()

# Prepare X and Y
x = []
y = []
print()
for group in training_set:
    for features in training_set[group]:
        print(group, features)
        x.append(features)
        y.append(group)

"""
K Neighbors Classifier
Simple algorithm which works on measuring distance between testing
sample to training samples and determine k nearest neighbors.
"""
# Initialize and fit
clf = neighbors.KNeighborsClassifier()
clf.fit(x, y)

# Preprocess testing data
before = np.array(testing_set)
# Reshape your data using array.reshape(-1, 1) if your data has a single feature.
after = before.reshape(1, -1)

print("\nBefore:", before, before.shape)
print("After:", after, after.shape)

# Prediction
prediction = clf.predict(after)  # testing_set
print("\nprediction:", prediction)

exit(0)

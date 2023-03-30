"""
Traditional example dog vs cat classification
3 images of dogs and 3 images of cats; images have only 2 pixel values x, y (so we can easily plot them)
Import -> Initialize -> Fit(Train) -> Predict
https://medium.com/@parthvadhadiya424/hello-world-program-with-scikit-learn-a869beb55deb
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors  # import model building

# Python dict that contains two classes(dog, cat) with three samples each
# and two features (maybe height, width)
training_set = {'Dog': [[1, 2], [2, 3], [3, 1]], 'Cat': [[11, 20], [14, 15], [12, 15]]}

# Testing set that we will use for prediction
testing_set = [15, 20]

# PLOT
c = 'x'
for data in training_set:
    print(data)
    # print(training_set[data])

    for i in training_set[data]:
        plt.plot(i[0], i[1], c, color='c')

    c = 'o'
plt.show()

# Prepare X and Y
x = []
y = []
for group in training_set:

    for features in training_set[group]:
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
testing_set = np.array(testing_set)
testing_set = testing_set.reshape(1, -1)

# Prediction
prediction = clf.predict(testing_set)
print(prediction)

exit(0)

# import json

# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# load the saved model from a file
loaded_model = keras.models.load_model('my_model.h5')

# use the loaded model to make predictions
predictions = loaded_model.predict(x_test)

# Do you really expect me to look through 10K values?
# Print to text file
# with open('y_test.txt', 'w') as filehandle:
#     json.dump(y_test.tolist(), filehandle)
# with open('predictions.txt', 'w') as filehandle:
#     json.dump(predictions.tolist(), filehandle)

# Alright, what the hecc is going on here?
# print("test length: {}, type: {}, shape: {}".format(len(y_test), type(y_test), np.shape(y_test)))
# print("pred length: {}, type: {}, shape: {}".format(len(predictions), type(predictions), np.shape(predictions)))

# Messy Output
# for i in range(len(predictions)):
#     # print("Input: {}, True output: {}, Predicted output: {}".format(x_test[i], y_test[i], predictions[i]))
#     print("Predicted output: {}".format(predictions[i]))

predicted_labels = predictions.argmax(axis=1)

# GRAPH
# plot the true outputs vs. the predicted outputs
# plt.scatter(y_test, predictions)
# plt.scatter(y_test, predicted_labels)
# plt.xlabel("True outputs")
# plt.ylabel("Predicted outputs")
# plt.show()

# assuming y_test is the true labels and predicted_labels is the predicted labels
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predicted_labels))


# TODO: print("\nCONFUSION MATRIX")
# None of them work. It all started with a tensorflow version confusion that may or may not be.
# Also - could it be because I'm doing the sklearn one in the same program?


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def conf_mat1():
    # assuming y_test is the true labels and predicted_labels is the predicted labels
    cm = tf.math.confusion_matrix(y_test, predicted_labels)
    # calculate the F1-score for each class
    f1_scores = tf.math.reduce_mean(tf.math.f1_score(y_test, predicted_labels, axis=0))
    print("Confusion matrix:")
    print(cm.numpy())
    print("F1-scores for each class:")
    print(f1_scores.numpy())


def conf_mat2():
    # assuming y_test is the true labels and predicted_labels is the predicted labels
    cm = tf.math.confusion_matrix(y_test, predicted_labels)

    # calculate the F1-score for each class
    precision = tf.math.divide_no_nan(tf.linalg.diag_part(cm), tf.reduce_sum(cm, axis=0))
    recall = tf.math.divide_no_nan(tf.linalg.diag_part(cm), tf.reduce_sum(cm, axis=1))
    f1_scores = f1_score(precision, recall)
    print("Confusion matrix:")
    print(cm.numpy())
    print("F1-scores for each class:")
    print(f1_scores.numpy())


def conf_mat3():
    print("\nCONFUSION MATRIX")
    # assuming y_test is the true labels and predicted_labels is the predicted labels
    cm = tf.math.confusion_matrix(y_test, predicted_labels)
    # calculate the precision and recall for each class
    precision = tf.math.divide_no_nan(tf.linalg.diag_part(cm), tf.cast(tf.reduce_sum(cm, axis=0), tf.float32))
    recall = tf.math.divide_no_nan(tf.linalg.diag_part(cm), tf.cast(tf.reduce_sum(cm, axis=1), tf.float32))
    # calculate the F1-score for each class
    f1_scores = 2 * precision * recall / (precision + recall)
    print("Confusion matrix:")
    print(cm.numpy())
    print("F1-scores for each class:")
    print(f1_scores.numpy())


def conf_mat4(y_test):
    # assuming y_test is the true labels and predicted_labels is the predicted labels
    cm = tf.math.confusion_matrix(y_test, predicted_labels)

    # calculate the precision and recall for each class
    precision = tf.math.divide_no_nan(tf.linalg.diag_part(cm), tf.cast(tf.reduce_sum(cm, axis=0), tf.float32))
    recall = tf.math.divide_no_nan(tf.linalg.diag_part(cm), tf.cast(tf.reduce_sum(cm, axis=1), tf.float32))

    # calculate the F1-score for each class
    f1_scores = 2 * precision * recall / (precision + recall)
    # todo: heh?  not in loop?

    # convert y_test to float32
    y_test = tf.cast(y_test, tf.float32)

    # generate the classification report
    print(classification_report(y_test, predicted_labels))

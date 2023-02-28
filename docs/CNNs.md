## Image Classification

<img src="https://149695847.v2.pressablecdn.com/wp-content/uploads/2017/09/localizationVsDetection.png" width="600">

<br>
[simple-keras.py](../bear_training/classification/simple-keras.py) uses the **Keras library** to build a convolutional neural network (**CNN**) model to classify images of **handwritten digits** from the **MNIST** dataset.

### Ok, <i style="all:revert">what</i> CNN?

<!-- https://www.tutorialrepublic.com/faq/how-to-reset-or-remove-css-style-for-a-particular-element.php -->

The Convolutional Neural Network (CNN) used for image classification is commonly referred to as a **"ConvNet"** or "CNN architecture".

ConvNets were first introduced in the 1980s by Yann LeCun.

Different CNN architectures:

* **LeNet**
    * Simple convolutional neural network, developed for handwritten digit recognition.
    * "LeCun et al." 🇫🇷
* **AlexNet**
    * Object-detection, computer vision
    * Alex Krizhevsky 🇨🇦
    * Creator of the CIFAR-10 and CIFAR-100 datasets.
* **VGGNet**
    * Small 3x3 convolutional filters, known for its simplicity and strong performance.
    * "Visual Geometry Group", Oxford University 🏴󠁧󠁢󠁥󠁮󠁧󠁿🇬🇧
* **InceptionNet** (InceptionV3)
    * "Inception modules" allow for efficient use of computational resources.
    * Image analysis and object detection
    * Got its start as a module for GoogLeNet 🇺🇸
* **ResNet**
    * Deep Residual Networks (ResNets)
    * Uses residual connections to overcome the problem of vanishing gradients.
    * [What Does "Residual" Mean?](https://analyticsindiamag.com/why-resnets-are-a-major-breakthrough-in-image-processing/) Residual is nothing but the error.
    * Kaiming He 🇺🇸
    * <img src="https://em-content.zobj.net/thumbs/320/emojipedia/102/flag-for-california-usca_1f3f4-e0075-e0073-e0063-e0061-e007f.png" width="50"> 

<span style="color:#0000dd;font-size:larger;">See:</span> [ResNet50.md](Tensorflow/ResNet50.md)

### Layers

Both Conv2D and MaxPooling2D are **layers** commonly used in Convolutional Neural Networks (CNNs).

**Conv2D** is a type of convolutional layer.

**MaxPooling2D** is a type of pooling layer.

Both are fundamental components of CNNs.

### Pro-tip!

When you get:

```
"ModuleNotFoundError: No module named 'keras.models'; 'keras' is not a package"
```

<br>
<span style="color:red;font-size:larger">"Dr. Snoopy" says:</span>

The problem is that you have a file named **"keras.py"** and this shadows the real keras package.

Don't do that, never name a python script the same as a package.

Solution is to **rename your script.**

I knew that. 🤦‍♀️

## Output

10 epochs.

Loss. Accuracy. val\_loss. val\_accuracy.

Final Accuracy.

```c
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

Epoch 1/10
300/300 [==============================] - 8s 25ms/step - loss: 0.2728 - accuracy: 0.9240 - val_loss: 0.1027 - val_accuracy: 0.9719

Etc.

Epoch 10/10
300/300 [==============================] - 10s 34ms/step - loss: 0.0098 - accuracy: 0.9974 - val_loss: 0.0392 - val_accuracy: 0.9880

Accuracy: 98.80%
```

<br>

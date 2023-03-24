## Convolutional Neural Network (CNN)

Why would we use it?  What is it known for?  Why is it useful?

A CNN is a type of computer program that's designed to "look" at **pictures or videos** and figure out what's in them. It does this by breaking the image up into tiny pieces ("pixels") and analyzing each piece to see if it matches any patterns that it has learned from other pictures.

One of the main things that a CNN is known for is its ability to **recognize objects** in pictures.

For example, if you show a CNN a bunch of pictures of cats and a bunch of pictures of dogs, it can learn to tell the difference between them.

This can be really helpful in all sorts of situations, from identifying animals in wildlife photos to **detecting tumors** in medical images.

Another thing that makes CNNs really powerful is their ability to **"generalize"** what they've learned.

This means that if you show a CNN a new picture that it hasn't seen before, it can still use its knowledge of patterns to make an educated guess about what's in the picture.

For example, if a CNN has seen lots of pictures of cars, it can probably recognize a new picture of a car even if it's a different color or from a different angle.

So, the purpose of a CNN is to help computers understand pictures and videos in a way that's similar to how humans do it.

By breaking images down into tiny pieces and looking for patterns, CNNs can recognize objects and make sense of complex visual information.


## Image Classification

<img src="https://149695847.v2.pressablecdn.com/wp-content/uploads/2017/09/localizationVsDetection.png" width="600">

<br>

[simple-keras.py](../bear_training/classification/keras-mnist-class.py) uses the **Keras library** to build a convolutional neural network (**CNN**) model to classify images of **handwritten digits** from the **MNIST** dataset.

<!--<i style="all:revert">What</i>-->

<!-- https://www.tutorialrepublic.com/faq/how-to-reset-or-remove-css-style-for-a-particular-element.php -->

The Convolutional Neural Network (CNN) used for image classification is commonly referred to as a **"ConvNet"** or "CNN architecture".

## Historical stuff

ConvNets were first introduced in the 1980s by Yann LeCun.

Different CNN architectures:

* **LeNet**
    * Simple convolutional neural network, developed for handwritten digit recognition.
    * "**Le**Cu**n et** al." 🇫🇷
    * "LeNet" is typically pronounced as "luh-net."
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

## Misc

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

<br>

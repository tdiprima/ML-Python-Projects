## What is ResNet50

<span style="color:#0000dd;font-size:larger;">A type of deep neural network used for image 
recognition.</span>

<!--<img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png" width="600">-->

A good image to remind yourself that ResNet50 is used for image recognition is a photograph or an image with multiple objects, such as a group of people, animals, or objects, with ResNet50's top predictions for each object's label shown alongside the object. This would emphasize ResNet50's ability to identify and classify objects in an image.

<img src="https://azati.ai/wp-content/uploads/2020/04/object-detection-800x400-1.jpg" width="600">

Alternatively, you could also use an image of the ResNet50 architecture diagram, which shows the deep residual layers used to train the model. The diagram usually has multiple layers and pathways, which are designed to help the model better recognize and classify different objects within an image. This would emphasize the technical details of ResNet50's architecture and how it enables the model to perform image recognition tasks.

<!--<img src="https://i.ytimg.com/vi/mGMpHyiN5lk/maxresdefault.jpg" width="600">-->

<img src="https://miro.medium.com/max/1200/0*tH9evuOFqk8F41FG.png" width="600">

The "50" in ResNet50 means that **the network has 50 layers.**

The more layers a network has, the more complex features it can learn, which can improve its ability to recognize images.

<span style="color:#0000dd;font-size:larger;">How many layers do they normally have?</span>

Between 3 and 20 layers.

## Residual connections <span style="font-size:27px;">🚇</span>

These connections allow the network to **"skip"** over some of the layers, which can help prevent the vanishing gradient problem. <span style="font-size:27px;">💨</span>

This problem can occur in very deep networks, where the **gradients** used to update the weights of the network become very small &ndash; **¡pequeño!** &ndash; making it difficult for the network to learn.

By using residual connections, ResNet50 can maintain a **strong gradient signal** 📶 throughout the network, making it easier to train and allowing it to learn more complex features.

This makes **ResNet50** a very powerful tool for **image recognition** tasks.

## Other DNNs

1. VGG16
1. Inception-v3
1. MobileNet
1. AlexNet
1. DenseNet
1. GoogLeNet
1. ResNet18
1. EfficientNet
1. YOLOv3
1. U-Net

<span style="color:#0000dd;font-size:larger;">See:</span> [CNNs.md](../CNNs.md)

<br>

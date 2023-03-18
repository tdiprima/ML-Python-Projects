## FashionMNIST dataset

When I print out train\_data[0], it gives me: Image shape: torch.Size([1, 28, 28]), tensor, 9.  My question is... nine what?

In the context of the `torchvision.datasets.mnist.FashionMNIST` dataset, the integer value `9` that you are seeing when you print `train_data[0]` refers to the class label of the image.

Each image in the FashionMNIST dataset is assigned a class label, which is an integer between 0 and 9 that corresponds to a specific article of clothing. The mapping of class labels to clothing types is as follows:

```makefile
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```

In this case, since the value of 9 is being printed, it means that the image in `train_data[0]` represents an ankle boot.

The `Image shape: torch.Size([1, 28, 28])` part of the printout indicates the shape of the image tensor, which has one channel (grayscale), a height of 28 pixels, and a width of 28 pixels. The `tensor` part simply indicates that the value being printed is a PyTorch tensor object.

## List attributes of dataset

I set up my training data; it's of type torchvision.datasets.mnist.FashionMNIST.

It has a `.classes` attribute.  I wanted to see what the other attributes were.  So I did `vars(train_data)`.  I don't even see ".classes" in the list.  Why?

The **`vars()`** function returns a dictionary containing the attributes and their values of an object. However, some attributes may not be included in the dictionary if they are implemented using properties or other methods that are not stored as regular instance variables.

In the case of `torchvision.datasets.mnist.FashionMNIST`, the `.classes` attribute is **implemented as a property**, which means that it is **calculated dynamically** when accessed and is not actually stored as an instance variable. Therefore, it will not appear in the dictionary returned by `vars(train_data)`.

Instead of using `vars()`, you can use the **`dir()`** function to **list all the attributes** of an object, including **properties:**

```python
print(dir(train_data))
# OR
import pprint as pp
pp.pprint(dir(train_data))
```

This will give you a list of all the **attributes** and **methods** of `train_data`, including the `classes` property. You can then access it using `train_data.classes`.


## Grayscale purple 💬

If the image is grayscale, then how come when I visualize it in matplotlib, it has a purple background?

When you visualize a grayscale image in Matplotlib, it is usually displayed with a colormap, which maps the pixel values of the image to different colors. By default, Matplotlib uses the **'viridis' colormap**, which goes from dark purple for low pixel values to bright yellow for high pixel values.

In your case, it's possible that the background of your grayscale image has a low pixel value, causing it to be mapped to the dark purple color in the colormap. The ankle boot in the image likely has higher pixel values, which are mapped to lighter colors such as yellow or white.

If you want to display the grayscale image without any colormap, you can use the `cmap='gray'` parameter when calling `imshow()` in Matplotlib:

```python
import matplotlib.pyplot as plt

# assuming image is a tensor containing the grayscale image data
plt.imshow(image[0], cmap='gray')
plt.show()
```

This will display the grayscale image with a simple black and white color scheme.

<MARK>**NOTE!**</MARK>

You don't have to "squeeze" (`image.squeeze()`), but you do have to `image[0]`.

One or the other.

Also: `plt.title(class_names[label]);` to get the name that's associated with that label.


<span style="color:#00ff00;font-weight:bold;font-size:24pt;">Viridis</span>

Green!  Young, fresh, lively, youthful.

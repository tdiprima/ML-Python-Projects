## Setting number of workers

<mark>**TL;DR: This code sux, and it literally takes forever. Use the PyTorch one.**</mark>

Removed code for non-confusion.  And I'm assuming I mean <a href="../../../PyTorch/image_segmentation/unet-carvana/using-model-2.ipynb">unet-carvana</a> or any of those in `image_segmentation`.

[Image Seg Carvana](https://towardsdatascience.com/image-segmentation-predicting-image-mask-with-carvana-data-32829ca826a0)

[GitHub](https://github.com/ZeeTsing/Carvana_challenge.git)

## Take part of the dataset

What is this doing?

```python
all_images = os.listdir("/path/to/data")
to_train = 1  # all
# to_train = 0.1  # some
total_train_images = all_images[:int(len(all_images) * to_train)]
```

* This code is reading the names of all files in a directory located at "/path/to/data" and storing them in the list `all_images`.

* Then, it is setting a variable `to_train` to 1, indicating that we want to use all the available images for training.

* Next, it selects a subset of the images to use for training by slicing the `all_images` list.

The slice includes the first `int(len(all_images) * to_train)` elements of the list.

The `len(all_images)` function returns the total number of images.

`to_train` is used to specify what fraction of those images should be used for training.

The `int()` function is used to convert the resulting fraction to an integer, as slicing requires integer indices.

The resulting subset of images is stored in the `total_train_images` list. This list can then be used for **training a machine learning model.**

## Double-slashes

What does it mean when someone puts double slashes, like this: `1918//1920`?

When you use double slashes `//` in Python, it performs integer division and returns the floor value of the quotient.

<span style="color:#0000dd;font-size:larger;">quotient: a result obtained by dividing one quantity by another.</span>

For example, if you have `1918//1920`, the result will be `0`. This is because the quotient of dividing 1918 by 1920 is less than 1, so the floor value of this quotient is 0.

Another example is `10//3`, which will give you `3`. This is because the quotient of dividing 10 by 3 is 3 with a remainder of 1. The double slashes `//` will return the floor value of this quotient, which is 3.

On the other hand, if you use a single slash `/`, it will perform normal division and return the exact quotient. For example, `10/3` will give you `3.33333333`.

## Integer division

What is this trying to tell me: "1918//1920 divisive by 64"?

Without additional context, it is difficult to determine the exact meaning of the statement "1918//1920 divisive by 64". However, here is a possible interpretation:

The double-slash operator `//` is being used to perform integer division of `1918` by `1920`, which results in a quotient of `0`. The statement "divisive by 64" may mean that the quotient is a multiple of `64`, or that it can be expressed as the product of `64` and some integer value.

To verify whether `1918//1920` is indeed divisible by `64`, you could perform the division and check whether the remainder is zero:

```python
quotient = 1918//1920
if quotient % 64 == 0:
    print("1918//1920 is divisible by 64")
else:
    print("1918//1920 is not divisible by 64")
```

This code will output "1918//1920 is not divisible by 64" because the quotient is `0` and therefore cannot be expressed as a multiple of `64`.

## Random augmentation

When we're working with images in a computer, we often want to make the images look better or easier to understand. One way to do this is by using random augmentation.

Random augmentation is when we take an image and make small changes to it in a random way. These changes can be things like rotating the image a little bit, making it a little bit bigger or smaller, or changing the brightness or color of the image. By making these small changes, we can create lots of new images that are similar to the original, but not exactly the same.

Why would we want to do this? Well, it turns out that when we're teaching a computer to recognize things in images (like cats or dogs), it's helpful to have lots of different images for the computer to learn from. By using random augmentation to create new images, we can give the computer a bigger set of images to learn from, which can make it better at recognizing things in the real world.

So that's what random augmentation is: it's a way to create lots of new images that are similar to the original, but not exactly the same. By doing this, we can help computers get better at recognizing things in images!

## Conv2D input_layer

```python
conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
```

What's that `(input_layer)` at the end?

In the U-Net model, the `(input_layer)` at the end of the `Conv2D` call is not multiplying it. Instead, it is passing the `input_layer` tensor to the `Conv2D` layer as an input.

The `Conv2D` layer is a type of layer in a neural network that applies a set of filters to the input tensor to extract features from the input.

* In the code you provided, `filters` is the number of filters to use.

* `(3, 3)` is the size of each filter. 

* `padding='same'` specifies that the output of the convolution operation should have the same spatial dimensions as the input.

* Finally, `activation='relu'` specifies that the rectified linear unit (ReLU) activation function should be applied to the output of the convolution operation.

The `input_layer` tensor represents the input to the neural network. It is typically a multi-dimensional array that contains the pixel values of an image or some other type of data. In the U-Net model, the `input_layer` tensor is passed as input to the first `Conv2D` layer, which applies a set of filters to it to extract low-level features from the input image.

So in short, `(input_layer)` at the end of the `Conv2D` call is not multiplying it, but rather passing the `input_layer` tensor as input to the `Conv2D` layer to extract features from it.

## Yield

[What does the "yield" keyword do in Python?](https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python#231855)

`yield` is a keyword that is used like `return`, except the function will **return a generator.**

Your function will **return a huge set of values** that you will only need to **read once.**

# ¿Problemas?

## TensorFlow Memory Usage Warning

<span style="color:red;">Allocation of 1207959552 exceeds 10% of free system memory.</span>

Would reducing the batch size help?

[post](https://stackoverflow.com/questions/50304156/tensorflow-allocation-memory-allocation-of-38535168-exceeds-10-of-system-memor)

[another post](https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/42121886#42121886)

The warning you are seeing in your TensorFlow program is indicating that the system memory usage has exceeded the limit set by TensorFlow. This can happen if your program is allocating more memory than the system can provide.

Reducing the batch size can help to reduce the memory usage and alleviate the warning. However, it may not always be the best solution as it can also impact the accuracy of your model.

Before you reduce the batch size, you may want to check if there are any other parts of your program that could be causing the high memory usage. Here are a few things to consider:

1. Check if there are any other programs running that are using a lot of memory. You may want to close them or reduce their memory usage.

2. Check if there are any large variables being created in your program that are not needed. You may want to remove them or store them in a file instead.

3. Check if you are using any unnecessary data types or operations that are causing high memory usage. For example, using float64 instead of float32 can double the memory usage.

If you have done all of the above and are still seeing the warning, then reducing the batch size may be a reasonable solution. However, it is important to keep in mind that reducing the batch size can also impact the convergence rate and the quality of your model's predictions.

Therefore, you may want to experiment with different batch sizes to find a balance between memory usage and model accuracy.

## Placeholder Error

Just a warning, it even says you can ignore it:

<span style="color:red;">You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,?,?,?]</span>

<span style="color:green;">In the brain tumor one, it said shape [?,73,73,64].  There you go.</span>

[post](https://de-de.facebook.com/groups/TensorFlowKR/posts/753755691632158/)

Jung Hye-min feels finished.<br>
18. September 2018

I get an error like this.  If you take a picture of the input that goes into the feed dict, it comes out as 50,256,256,3.

(They're getting the error at `sess.run()`, but I'm not using it.)

**Answer:** If only the global variable is initialized, it will probably work.

(Maybe it has to do with the "None" batch size.  Batch size actually does work, but IDK why it's saying None.)

## Placeholder Warning Troubleshooting

<span style="color:red;">You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32.</span>

There could be several reasons why you are getting this error, even if you are not using `tf.placeholder()` directly in your code. Here are some possible explanations:

1. You may have used a pre-trained model or a code snippet that includes `tf.placeholder()`, and you are not providing the required input to the placeholder when running the model. In this case, you may need to inspect the code or documentation for the pre-trained model or snippet to understand what input is expected.

2. You may have defined a `tf.Variable()` or `tf.constant()` with the same name as the placeholder tensor, and the error message is referring to that instead. For example, if you have a variable named `Placeholder` and you try to run a part of the code that expects a placeholder tensor with the same name, you could get this error. To fix this, you should rename the variable to something else.

3. You may be using a library or function that internally uses `tf.placeholder()` without exposing it in the API. In this case, you may need to inspect the source code of the library or function to see if there is any mention of placeholders, and try to understand how they are being used.

4. It is also possible that the error message is misleading, and the issue is actually related to a different part of your code. To troubleshoot this, you can try to narrow down the scope of the problem by commenting out parts of your code and running it again to see if the error goes away.

## Found untraced functions such as...

I often see this warning when saving custom models such as graph NNs. You should be good to go as long as you don't want to access those non-callable functions.

However, if you're annoyed by this big chunk of text, you can suppress this warning by adding the following at the top of the code.

```py
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
```

[another answer](https://stackoverflow.com/questions/65697623/tensorflow-warning-found-untraced-functions-such-as-lstm-cell-6-layer-call-and#73229760)

<br>

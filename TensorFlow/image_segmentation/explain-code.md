## Setting number of workers

<!--
https://towardsdatascience.com/image-segmentation-predicting-image-mask-with-carvana-data-32829ca826a0
https://github.com/ZeeTsing/Carvana_challenge.git
-->

In TensorFlow, the number of workers is typically set when using a distributed computing approach, such as in a multi-node or multi-GPU environment. The exact method for setting the number of workers may vary depending on the specific approach being used, but generally it involves setting the appropriate environment variables or passing arguments to the TensorFlow program.

Here is an example of how to set the number of workers in TensorFlow when using the `tf.distribute` module for distributed training:

```python
import tensorflow as tf

# Set the number of worker nodes
num_workers = 4

# Define the distribution strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(num_workers=num_workers)

# Build and compile the model within the distribution strategy scope
with strategy.scope():
    model = tf.keras.Sequential([...])
    model.compile([...])
```

In this example, we set the `num_workers` variable to 4 to indicate that we have **4 worker nodes.**

We then create a `MultiWorkerMirroredStrategy` object with this value, which will enable us to distribute the training across multiple worker nodes.

Finally, we **build and compile** the model within the distribution strategy scope using `strategy.scope()`, which will ensure that the model is replicated and trained in parallel across all the worker nodes.

## Nope.

`_CollectiveAllReduceStrategyExperimental.__init__()` got an unexpected keyword argument 'num_workers'

The CollectiveAllReduceStrategyExperimental is a TensorFlow strategy that can be used for distributed training. However, it does not have a num_workers argument in its constructor.

The CollectiveAllReduceStrategyExperimental strategy uses all available devices on a single machine to perform synchronous training. When using this strategy, there is no need to specify the number of workers since it automatically uses all available devices.

Here's an example of how to use CollectiveAllReduceStrategyExperimental:

```python
import tensorflow as tf

# Create the strategy
strategy = tf.distribute.experimental.CollectiveAllReduceStrategy()

# Define and compile the model within the strategy scope
with strategy.scope():
    model = tf.keras.Sequential([...])
    model.compile([...])
```

<span style="color:red;">It didn't like this one, either.</span>

In this example, we create an instance of the CollectiveAllReduceStrategyExperimental strategy, which will use all available devices on a single machine for training. We then define and compile the model within the strategy scope, which will ensure that the model is replicated and trained in parallel across all available devices.

## Take part of the dataset

What is this doing?

```python
all_images = os.listdir("/path/to/data")
to_train = 1  # all
# to_train = 0.1  # some
total_train_images = all_images[:int(len(all_images) * to_train)]
```

This code is reading the names of all files in a directory located at "/path/to/data" and storing them in the list `all_images`.

Then, it is setting a variable `to_train` to 1, indicating that we want to use all the available images for training.

Next, it selects a subset of the images to use for training by slicing the `all_images` list.

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

## Selecting rows based on unique values

```python
import pandas as pd

METADATA_CSV = pd.read_csv("path/to/file")
TRAIN_MASKS_CSV = pd.read_csv("path/to/file1")

train_metadata_csv = METADATA_CSV.loc[TRAIN_MASKS_CSV['id'].unique(), :]
```

This line of code is selecting rows from the `METADATA_CSV` DataFrame based on the unique values in the `id` column of the `TRAIN_MASKS_CSV` DataFrame.

Specifically, `TRAIN_MASKS_CSV['id'].unique()` returns an array of unique values in the `id` column of the `TRAIN_MASKS_CSV` DataFrame.

This array is then used to index into the `METADATA_CSV` DataFrame using the `.loc` accessor, which allows you to select rows and columns by label or boolean mask.

The `:` in the second argument of the `.loc` accessor means to **select all columns** in the DataFrame.

The first argument `TRAIN_MASKS_CSV['id'].unique()` **specifies the rows to select** based on the unique values in the `id` column of `TRAIN_MASKS_CSV`.

So overall, the line of code is selecting all rows from `METADATA_CSV` whose `id` column matches a **unique value** in the `id` column of `TRAIN_MASKS_CSV`.

## Like sql

Something like that.

```sql
select * from table as t where t.id = (select distinct id from blah)
```

## Mammoth pandas statement

Let's say the result of that goes into a variable called `train_metadata_csv`.  What does this do? 

```py
pd.DataFrame.from_dict(dict(Counter(train_metadata_csv.make)), orient='index').reset_index().rename(columns={'index': 'make', 0: 'make_count'})
```

This line of code generates a Pandas DataFrame with two columns: `make` and `make_count`.

The `make` column contains the **unique values** in the `make` column of the `train_metadata_csv` DataFrame.

The `make_count` column contains the **count** of each unique value in the `make` column.

Here's how the line of code works step by step:

* `Counter(train_metadata_csv.make)` **creates a dictionary** where the keys are the unique values in the make column of train_metadata_csv, and the values are the count of each unique value in the make column.

* `dict(Counter(train_metadata_csv.make))` **converts this dictionary** into a regular Python dictionary.

* `pd.DataFrame.from_dict(dict(Counter(train_metadata_csv.make)), orient='index')` **converts the dictionary** into a Pandas DataFrame where the keys are used as the row labels and the counts are the values in a single column.

* `.reset_index()` **resets the row index** of the DataFrame and moves the current index (which is the unique values in the `make` column) into a new column called `index`.

* `.rename(columns={'index': 'make', 0: 'make_count'})` **renames the index column** to `make` and renames the single value column to `make_count`.

Overall, the line of code generates a DataFrame with two columns: `make` and `make_count`. The `make` column contains the unique values in the `make` column of `train_metadata_csv`, while the `make_count` column contains the count of each unique value in the `make` column.

## ¿Problemas?

### cpu_allocator_impl.cc:83] Allocation of 1207959552 exceeds 10% of free system memory.

https://stackoverflow.com/questions/50304156/tensorflow-allocation-memory-allocation-of-38535168-exceeds-10-of-system-memor

https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/42121886#42121886


Just a warning, it even says you can ignore it:

### You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,?,?,?]

[post](https://de-de.facebook.com/groups/TensorFlowKR/posts/753755691632158/)

Jung Hye-min feels finished.<br>
18. September 2018

I get an error like this.  If you take a picture of the input that goes into the feed dict, it comes out as 50,256,256,3.

(They're getting the error at `sess.run()`, but I'm not using it.)

**Answer:** If only the global variable is initialized, it will probably work.

(Maybe it has to do with the "None" batch size.  Batch size actually does work, but IDK why it's saying None.)

### You must feed a value for placeholder tensor 'Placeholder/\_0' with dtype int32

I'm not using tf.placeholder()!

<!--
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 24). These functions will not be directly callable after loading.
-->
<br>

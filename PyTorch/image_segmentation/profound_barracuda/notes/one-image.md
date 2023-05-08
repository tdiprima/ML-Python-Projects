## Run 1 image through U-Net

**Load** the image: You can use a library like OpenCV or PIL to load the image into memory.

### Pillow

```py
from PIL import Image
from tensorflow.keras.preprocessing import image

test_image = Image.open("alejandra.jpg")

print(test_image.size)  # (750, 1294)
```

<br>

**Preprocess** the input: U-Net expects input images to be of a certain size and format. You may need to resize the image, convert it to grayscale or RGB, and normalize its pixel values to a range between 0 and 1.

```py
# num batches = 1
img = image.img_to_array(test_image)  # (1294, 750, 3)
img = img.reshape((1,) + img.shape)   # (1, 1294, 750, 3)
```

<br>

Pass the image through the **network:** U-Net consists of an encoder, a decoder, and a skip connection that connects the encoder and decoder. You can use a deep learning framework like TensorFlow or PyTorch to load a pre-trained U-Net model, pass the preprocessed image through the model, and get the output.

**Postprocess** the output: The output of U-Net is typically a probability map that assigns a probability to each pixel indicating the likelihood of it belonging to a certain class. You may need to threshold the probability map to generate a binary segmentation mask.

## CV2 + TensorFlow ⏳

```py
import cv2
import numpy as np
import tensorflow as tf

# Load the image
image = cv2.imread("alejandra.jpg")

# resize the image
image = cv2.resize(image, (256, 256))

# convert it to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# normalize its pixel values to a range between 0 and 1
image = image / 255.0

# expand image dimensions
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

# Load the model
model = tf.keras.models.load_model("unet.h5")

# Pass the image through the model
output = model.predict(image)

# Post process the output
mask = np.squeeze(output) > 0.5
mask = np.uint8(mask) * 255

# Save the output
cv2.imwrite("mask.jpg", mask)
```

## What the hell is this?

```py
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)
```

<br>

When you read an image in Python with OpenCV (`cv2`), it will typically have a shape of `(height, width, channels)`.

<mark>`np.expand_dims`</mark> is used to add additional dimensions to the image array.  <span style="color:#0000dd;">(Obviously...)</span>

- `np.expand_dims(image, axis=-1)` adds a new dimension at the end of the array (i.e., along the last axis), effectively converting the shape of the image from `(height, width, channels)` to `(height, width, channels, 1)`. This is typically done to indicate that the image has a single channel, even if it was originally a color image with three channels.

- `np.expand_dims(image, axis=0)` adds a new dimension at the beginning of the array (i.e., along the first axis), effectively converting the shape of the image from `(height, width, channels, 1)` to `(1, height, width, channels, 1)`. This is typically done to indicate that the image is part of a batch of images, even if there is only one image in the batch.

The reason for using `axis=-1` in the first call to `np.expand_dims` is that it ensures the new dimension is added to the end of the array, which is where the channel dimension is located. Using `axis=0` in the second call ensures that the new dimension is added at the beginning of the array, which is where the batch dimension is typically located in deep learning frameworks.

Overall, this code is often used when processing images in batches, as it allows for a consistent format across all images in the batch.

## What?

<span style="color:#0000dd;">I get that the dimensions would be (batch_size, height, width, channels), but... when we add a dimension to the end, and the value is 1, I don't understand why we need that if we already have "channels".  Is "channels" 3, and then the 1 means - it's 3, but we're making it 1 (grayscale)?</span>

You are correct that adding a dimension with a value of 1 to the end of the array, as done with `np.expand_dims(image, axis=-1)`, does not change the number of channels in the image. However, this operation can be useful in certain cases.

For example, when working with deep learning frameworks that expect a 4D tensor as input (i.e., with a shape of `(batch_size, height, width, channels)`), it's important to have a consistent format across all images, even if they have different numbers of channels. By adding a new dimension with a value of 1, we effectively "pad" the channel dimension to 1, ensuring that the resulting tensor has a consistent shape.

In the case of grayscale images, `channels` is already 1, so adding another dimension with a value of 1 does not change the number of channels. However, it does ensure that the resulting tensor has a shape of `(height, width, channels, 1)` (4D), which is compatible with the expected input shape of many deep learning frameworks.

To summarize, adding a dimension with a value of 1 to the end of an image array does not change the number of channels in the image, but it can be useful for ensuring a consistent tensor shape when working with deep learning frameworks.

## CV2 + PyTorch 🔦

In PyTorch, you can pass the preprocessed image through the model and get the output...

1. **Load** the pre-trained U-Net model: You can load a pre-trained U-Net model using the `torch.load()` function.

2. Set the model to **evaluation** mode: This is important because it will disable dropout and batch normalization layers, which are only used during training.

3. **Convert** the image to a PyTorch tensor: You can use the `torch.from_numpy()` function to convert the preprocessed image to a PyTorch tensor.

4. Pass the tensor through the **model:** You can use the `.forward()` method of the model to pass the tensor through the model and get the output.

```py
import cv2
import numpy as np
import torch

# Load the image
image = cv2.imread("alejandra.jpg")

# Preprocess the input (same as before)
image = cv2.resize(image, (256, 256))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image / 255.0
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=1)

image_tensor = torch.from_numpy(image).float()

# Load the pre-trained U-Net model
model = torch.load('unet.pt')

# Set the model to evaluation mode
model.eval()

# Pass the tensor through the model
with torch.no_grad():
    output_tensor = model.forward(image_tensor)

# Convert the output tensor to a numpy array
output = output_tensor.numpy()

# Post process the output (same as before)
mask = np.squeeze(output) > 0.5
mask = np.uint8(mask) * 255

# Save the output
cv2.imwrite("mask.jpg", mask)
```

## Why was expand_dims -1 and 0 before, but it's 0 and 1 now?

The difference in the usage of axis values between the TensorFlow and PyTorch scripts can be attributed to the differences in their **default data formats.** 

In **TensorFlow**, the default data format for image tensors is **`NHWC`** (batch, height, width, channels).

In **PyTorch**, the default data format is **`NCHW`** (batch, channels, height, width).

In the TensorFlow script, `np.expand_dims(image, axis=-1)` adds an extra dimension at the end of the tensor (i.e., the channel dimension), whereas `np.expand_dims(image, axis=0)` adds a batch dimension at the beginning of the tensor. 

In the PyTorch script, `np.expand_dims(image, axis=0)` still adds a batch dimension at the beginning of the tensor. However, since the default data format for PyTorch is `NCHW`, the next dimension to add is the channel dimension, which is done using `np.expand_dims(image, axis=1)`.

The use of `-1` in TensorFlow is a shorthand for indicating the last dimension of the tensor, which is typically the channel dimension in image data.

However, in the PyTorch script, the channel dimension is ***second***, hence `1` is used instead of `-1`.

## Pre-trained models

Pre-trained models are typically trained on large datasets and can be fine-tuned on your specific task or used directly for inference.

You can download the pre-trained U-Net model in PyTorch from the torchvision library:

```py
import torchvision.models as models

model = models.segmentation.fcn_resnet101(pretrained=True)
```

There are many online repositories that provide pre-trained models for various deep learning tasks, including image segmentation.

For example, you can download a pre-trained U-Net model for biomedical image segmentation from the U-Net Zoo repository, or you can download pre-trained models from the Model Zoo in TensorFlow Hub.

When using a pre-trained model, it's important to make sure that the model is trained on data that is similar to your task.

<br>

## Tuple Unpacking

```py
img, label = test_data[0][:10]
```

*It looks like it's getting the first 10 elements from the column in the zeroth row... but that makes no sense.  And how do we know we're getting an image and a label from that?*

It's likely that `test_data` is a dataset object and 

`test_data[0]` returns the first element of the dataset, 

which is a tuple containing an image and its corresponding label. 

The code is then using **tuple unpacking** to assign the first 10 elements of the image and label to the variables `img` and `label`, respectively.

To clarify, `test_data[0]` likely returns a tuple of the form `(image, label)`, where `image` is a tensor representing the image data, and `label` is an integer representing the class label of the image.

`img, label = test_data[0][:10]` then takes the first 10 elements of this tuple and assigns them to the variables `img` and `label`, respectively.

<span style="color:#0000dd;font-size:larger;">Oh.</span>

However, this code may not work as intended if the size of the image tensor is less than 10 elements, so it's important to verify that the code is functioning as expected.

## Describe what the result of test_data[0][:10] might look like for a typical PyTorch dataset.

*I wanted an HTML table, but it couldn't do that.*

<span style="color:red;font-size:larger;">See: display_images.ipynb</span>

Assuming `test_data` is a PyTorch dataset object that contains images and their corresponding labels, `test_data[0]` returns a tuple containing the image and label for the first example in the dataset.

For example, let's say that the image is a 3x32x32 tensor (3 channels, 32x32 pixels), and the label is an integer. Then, test_data[0] might look like this:

```py
(tensor([[[0.0118, 0.0118, 0.0118, ..., 0.0078, 0.0078, 0.0078],
          [0.0118, 0.0118, 0.0118, ..., 0.0078, 0.0078, 0.0078],
          [0.0118, 0.0118, 0.0118, ..., 0.0078, 0.0078, 0.0078],
          ...,

# A whole bunch of code, and then it konks out.
```

<br>

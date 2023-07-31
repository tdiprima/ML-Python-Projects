## Plot multiple images in one figure window

```python
import matplotlib.pyplot as plt
import numpy as np

# Assume you have a list of image data, where each image is a 2D numpy array
image_list = [np.random.rand(10, 10), np.random.rand(10, 10), np.random.rand(10, 10)]

# Create a figure object and an array of axes objects to plot the images
fig, axs = plt.subplots(1, len(image_list), figsize=(10, 5))

# Iterate through the list of images and plot each image in a separate axis object
for i, image in enumerate(image_list):
    axs[i].imshow(image)

# Set the title and axis labels for the figure
fig.suptitle('List of Images')
fig.tight_layout()
plt.show()
```

<br>
First, we create a list of 3 sets of image data.

Then, we create a `figure` object and an array of axes objects using `subplots()`.

We set the `figsize` parameter to control the size of the figure window.

Next, we use a for loop to iterate through the list of images and plot each image in a separate axis object using `imshow()`.

We use `enumerate()` to get both the index and the value of each image in the list.

Finally, we set the title and axis labels for the figure using `suptitle()` and `tight_layout()`.

We then call `show()` to display the figure window with all the plotted images.

## Figsize

The figsize parameter is used to set the size of the figure in **inches**, not rows and columns.

`figsize=(10, 5)` sets the width of the figure to 10 inches and the height to 5 inches.

The **number of rows and columns** of subplots is set by the arguments passed to the **`subplots()`** method.

We passed `1` for the number of rows and `len(image_list)` for the number of columns, so that each image will be plotted in a separate column.

## Grid

If you wanted to plot the images in a grid of, say, 2 rows and 2 columns, you could change the `subplots()` line to:

```python
fig, axs = plt.subplots(2, 2, figsize=(10, 5))
```

This would create a figure with 2 rows and 2 columns of subplots, and you could plot up to 4 images in this grid.

## Flexible layout

<span style="color:#0000dd;">I suppose I could put a limit on the number of images to display, in order to fit the window we have set.  But let's suppose I don't know if we're going to display, for example, 4 images or 100 images.  Is there a method to expand and contract the window size accordingly?</span>

Yes, you can create a flexible layout for your figure that can automatically adjust its size based on the number of images you want to display.

```python
import matplotlib.pyplot as plt
import numpy as np

image_list = [np.random.rand(10, 10) for _ in range(50)]

# Calculate the number of rows and columns needed to display all the images
num_images = len(image_list)
num_rows = int(np.ceil(np.sqrt(num_images)))
num_cols = int(np.ceil(num_images / num_rows))

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

for i, image in enumerate(image_list):
    row_idx = i // num_cols  # row index, floor division
    col_idx = i % num_cols   # col index, modulus
    axs[row_idx, col_idx].imshow(image, cmap='gray')

fig.suptitle('List of Images')
fig.tight_layout()
plt.show()
```

In this example, we start by assuming that we have a list of image data, where each image is a 2D numpy array. We then calculate the number of rows and columns needed to display all the images. We use `np.ceil()` function to round up the number of rows and columns to ensure that we can fit all the images.

Next, we create a `figure` object and an **array of axes objects** using the `subplots()` method. We pass `num_rows` and `num_cols` for the number of rows and columns, respectively, to create a flexible layout that can accommodate any number of images. We also set the `figsize` parameter to control the size of the figure window.

We then use a `for` loop to iterate through the list of images and plot each image in a separate axis object. We calculate the row and column index of each image using integer division and modulo operations, respectively. We use these indices to access the correct axis object in the axs array.

Finally, we set the title and axis labels for the figure using the `suptitle()` and `tight_layout()` methods, respectively. We then call the `show()` method to display the figure window with all the plotted images.

By using this approach, the figure window will **automatically adjust its size** based on the number of images you want to display, and you can plot any number of images without having to manually adjust the figure size.

## Floor division for row index

<span style="color:#0000dd;">What is "//"?</span>

The double forward slash `//` is the floor division operator. It is used to perform integer division, which means the result will be an **integer** without any fractional part.

In the given code snippet:

```python
row_idx = i // num_cols
```

The line of code calculates the row index based on the value of `i` and the number of columns specified by `num_cols`. It determines in which row the element represented by `i` would be located when arranging elements in a two-dimensional structure (like a grid or a matrix).

For example, let's say `i` is 7, and `num_cols` is 3. Then:

```py
row_idx = 7 // 3
```

The result of `7 // 3` is 2, as 7 divided by 3 equals 2 with a remainder of 1. So, `row_idx` will be assigned the value of 2. This means that element `i` would be in the third row (as indices are zero-based).

In summary, the code calculates the row index of an element when arranging elements in a two-dimensional structure with the given number of columns.

<br>

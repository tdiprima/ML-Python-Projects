## What is a "tensor" anyway?

A tensor is a mathematical object used to represent **multi-dimensional arrays** of **numerical data.**

In the context of machine learning and deep learning, tensors are used to represent the **input and output data**, as well as the **parameters** and intermediate **activations** of neural network models.

In a neural network, a tensor is typically represented as a multi-dimensional **array of numbers**, with each dimension corresponding to a **different feature or aspect** of the data.

For example, an **image** may be represented as a **3-dimensional tensor**, with the first two dimensions corresponding to the width and height of the image, and the third dimension representing the color channels (e.g., **<span style="color:red">red,</span> <span style="color:green">green,</span> and <span style="color:blue">blue</span>**).

Tensors are a fundamental concept in many areas of mathematics, including **linear algebra** and **calculus.**

They are also used extensively in **physics** and **engineering** to represent physical quantities such as vectors and matrices.

In **PyTorch**, tensors are the fundamental building blocks of neural networks and machine learning models, and the framework provides a wide range of **operations and functions** for working with tensors efficiently and effectively.

## Visual Representation

Let's say we have a **3-dimensional tensor** that represents a **collection of images**, each with dimensions of 32 pixels by 32 pixels and 3 color channels **(<span style="color:red">red,</span> <span style="color:green">green,</span> and <span style="color:blue">blue</span>).**

We can represent this tensor as a block of data with **three axes:**

```
+------------------------------------------+
|                  Tensor                  |
+------------------------------------------+
|         | Width | Height | Channels      |
|---------|-------|--------|--------------|
| Image 1 | 32    | 32     | 3            |
| Image 2 | 32    | 32     | 3            |
| ...     | ...   | ...    | ...          |
| Image N | 32    | 32     | 3            |
+------------------------------------------+
```

<br>

1. The first axis of the tensor represents the individual **images**
2. The second axis represents the **width** of each image
3. The third axis represents the **height**
4. **color channels** of each image.

Each cell of the tensor represents a numerical value, which could be a pixel intensity value or some other kind of **feature value.**

For example, if we wanted to represent a **grayscale image** instead of a color image, we could simply **remove the third axis** and represent each image as a 2-dimensional tensor with dimensions 32x32.

<br>

## Batch size = 32

It's common in machine learning to use a batch size of 32.  I noticed that batch sizes are usually in multiples of 8.  What's up with that?

<mark>**Multiples of 8**</mark> are commonly used as batch sizes because most modern CPUs and GPUs have vector units that process data in chunks of 8 or 16 elements at a time. This means that using a batch size that is a multiple of 8 can help to fully utilize the vector units, leading to faster processing times.

> "Training with large minibatches is bad for your health.  More importantly, it's bad for your test error." &ndash; Yann LeCun

### Simply put...

You know how computers use a bunch of 0's and 1's to represent everything, right? Well, some numbers are easier for computers to work with because they can be represented using just a few 0's and 1's. These numbers are called powers of 2, and they look like this: 2, 4, 8, 16, 32, 64, and so on.

When we're teaching a computer how to learn from data (like images, for example), we often give it batches of information to work on. The batch size is the number of things we give the computer at once to look at and learn from.

To make things easier for the computer, we often use batch sizes that are powers of 2 (like 8, 16, 32, 64, etc.). This is because computers can work with these numbers really quickly and efficiently.

### 256 and 512

In the same way, when we save images, we often make them a certain size that's a power of 2, like 256x256 or 512x512. This is because it's easier for the computer to process and manipulate images that are a power of 2 in size.

So, by using batch sizes and image sizes that are powers of 2, we can help the computer work faster and more efficiently, which is really helpful when we're working with a lot of data!

<span style="color:#0000dd;font-size:larger;">Okay. So then...</span>

## Python: Exponentiation

### 2 to the power of what is 256?

```py
import math

result = math.log2(256)
print(result)  # 8.0
```

<br>
So there's your 2, your 8, and your 256!

`log2()` gives you the exponent that 2 needs to be raised to in order to get 256.

And 512 would be...

```py
import math
math.log2(512)  # 9.0
```

### oy

## Input Image Dimensions

Good morning!  Machine learning problems sometimes have input images, which are usually `224 x 224`.  I thought everything had to be to the power of 2.  Why are the dimensions of the images 224 x 224?

So first of all, it's not a power of 2, nor is it a power of 8.

```py
import numpy as np

x = 224
y = 8

num_multiplications = np.log2(x/y)
print(num_multiplications). # 4.807354922057604
```

Rather:

```c
28 × 8 = 224
112 × 2 = 224
```

Good morning! The dimensions of 224 x 224 in machine learning image recognition models are not necessarily because they have to be to the power of 2. The dimensions of 224 x 224 were chosen because it is a standard size that has been found to work well for many image recognition tasks, and is large enough to capture relevant features while still being small enough to be processed efficiently.

In fact, there are many other commonly used image sizes in machine learning, such as 128 x 128 or 256 x 256, which are also not powers of 2. The important thing is that the image size is consistent across the training and testing data, so that the model can learn to recognize features and patterns in the images regardless of their size. Additionally, the use of GPUs for processing has made it possible to use non-power-of-2 sizes efficiently as well.

### How can I use numpy to figure out what 256 is a power of?

You can use the `numpy.log2()` function to determine whether 256 is a power of 2. If 256 is a power of 2, then the result of `numpy.log2(256)` will be an integer.

```py
import numpy as np

if np.log2(256).is_integer():
    print("256 is a power of 2")
else:
    print("256 is not a power of 2")
```

<br>

## Batch size 32, encore.

Good question!

The reason why batch sizes are typically in multiples of 8 is because of the way computers process data. Computers use a binary system to represent data, which means that everything is represented using combinations of 1s and 0s.

In a binary system, numbers that are a power of 2 are represented by a single bit with a value of 1 followed by zeros. For example, 2 is represented as 10, 4 as 100, 8 as 1000, and so on. This means that numbers that are a power of 2 are easy for computers to process, as they can be represented and manipulated using a small number of bits.

In machine learning, batch sizes are typically set to powers of 2 to take advantage of this fact. This allows the computer to process the data more efficiently, as the batch size can be represented using a small number of bits.

Additionally, multiples of 8 are commonly used as batch sizes because most modern CPUs and GPUs have vector units that process data in chunks of 8 or 16 elements at a time. This means that using a batch size that is a multiple of 8 can help to fully utilize the vector units, leading to faster processing times.

So, in summary, batch sizes are typically set to powers of 2 and multiples of 8 to take advantage of the way computers process data, and to fully utilize vector units in modern CPUs and GPUs.

*I feel like this is similar to how we save images in sizes of multiples of 256, or 512.* <- Yes.

## The long way around

2 to the power of what is 256?

You can use a loop to repeatedly multiply 2 by itself until you reach 256, and count the number of iterations it takes to get there.

```python
num = 2
count = 0
while num < 256:
    num *= 2
    count += 1

if num == 256:
    print("2 multiplied by itself", count, "times equals 256.")
else:
    print("2 cannot be multiplied by itself evenly to get 256.")
```

In this code, we start with the number 2 and keep multiplying it by itself until we reach 256. The count variable keeps track of the number of iterations it takes to get there.

If we reach 256, we print a message saying how many times we multiplied 2 by itself to get there. If we don't reach 256 exactly, we print a message saying that 2 cannot be multiplied by itself evenly to get 256.

Running this code will give you the output: "2 multiplied by itself 8 times equals 256."

### Yeah, but isn't there a "power of" operation or something...?

Yes, there is a power operator in Python, denoted by two asterisks `**`. You can use the power operator to find the exponent needed to raise 2 to get 256. Here's an example code:

```python
exponent = 0
while 2 ** exponent != 256:
    exponent += 1

print("2 raised to the power of", exponent, "equals 256.")
```

In this code, we start with an exponent of 0 and keep increasing it until we reach 256 when we raise 2 to the power of the exponent using the power operator `**`. Once we reach 256, we print a message saying what exponent was needed to get there.

Running this code will give you the output: "2 raised to the power of 8 equals 256."

<br>

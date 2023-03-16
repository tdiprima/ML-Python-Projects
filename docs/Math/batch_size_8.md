## Batch size = 32

It's common in machine learning to use a batch size of 32.  I noticed that batch sizes are usually in multiples of 8.  What's up with that?

Multiples of 8 are commonly used as batch sizes because most modern **CPUs and GPUs** have vector units that **process data in chunks of 8 or 16 elements at a time.** This means that using a batch size that is a multiple of 8 can help to fully utilize the vector units, leading to faster processing times.

<!--Good question!

The reason why batch sizes are typically in multiples of 8 is because of the way computers process data. Computers use a binary system to represent data, which means that everything is represented using combinations of 1s and 0s.

In a binary system, numbers that are a power of 2 are represented by a single bit with a value of 1 followed by zeros. For example, 2 is represented as 10, 4 as 100, 8 as 1000, and so on. This means that numbers that are a power of 2 are easy for computers to process, as they can be represented and manipulated using a small number of bits.

In machine learning, batch sizes are typically set to powers of 2 to take advantage of this fact. This allows the computer to process the data more efficiently, as the batch size can be represented using a small number of bits.

Additionally, multiples of 8 are commonly used as batch sizes because most modern **CPUs and GPUs** have vector units that **process data in chunks of 8 or 16 elements at a time.** This means that using a batch size that is a multiple of 8 can help to fully utilize the vector units, leading to faster processing times.

So, in summary, batch sizes are typically set to powers of 2 and multiples of 8 to take advantage of the way computers process data, and to fully utilize vector units in modern CPUs and GPUs.-->

<!--I feel like this is similar to how we save images in sizes of multiples of 256, or 512...
-->

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

<!--## The long way around

You can use a **loop** to repeatedly multiply 2 by itself until you reach 256, and count the number of iterations it takes to get there.

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

Running this code will give you the output: **"2 multiplied by itself 8 times equals 256."**

### Yeah, but isn't there a "power of" operation or something...?

Yes, there is a power operator in Python, denoted by two asterisks `**`. You can use the power operator to find the exponent needed to raise 2 to get 256. Here's an example code:

```python
exponent = 0
while 2 ** exponent != 256:
    exponent += 1

print("2 raised to the power of", exponent, "equals 256.")
```

In this code, we start with an exponent of 0 and keep increasing it until we reach 256 when we raise 2 to the power of the exponent using the power operator `**`. Once we reach 256, we print a message saying what exponent was needed to get there.

Running this code will give you the output: **"2 raised to the power of 8 equals 256."**-->

<br>

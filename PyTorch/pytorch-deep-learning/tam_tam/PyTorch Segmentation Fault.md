<span style="color:#0000dd;font-size:larger;">I'm using ReLU on the fashion mnist dataset.  But it's saying "31530 segmentation fault".  What's that supposed to mean?</span>

Here.  [StackOver](https://stackoverflow.com/questions/49414841/process-finished-with-exit-code-139-interrupted-by-signal-11-sigsegv#49414907)

"This indicates a bug in your program. In a Python program, this is either a bug in the interpreter or in an extension module being used (and the latter is the most common cause)."

## I'd like to solve the puzzle

<mark>I reduced code #2 from 1000 to 256 samples and it worked.</mark>

<mark>Another thing that can mess it up: IMPORT ORDER MATTERS.</mark>

I think this is what messed up #2.

* He wanted 1000 samples.
* But then he set # epochs to 100.
* **It shoulda been 1000.**
* And then instead of (if epoch % 10 == 0:), do (if epoch % 100 == 0:).

Now it's all good.  With 1000 samples.

Note!  I'll betcha if I play with the numbers, I could come up with more ideas; but we get the point, and it works.

OK &ndash; also, remember to put the `params` param; don't just leave it:

```python
# Create an optimizer
# optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.1)
```

## Some intel

A "segmentation fault" error typically indicates that your program is trying to access memory that it is not supposed to access. This can happen when you use an uninitialized pointer or array, access memory beyond the bounds of an array, or attempt to write to read-only memory, among other things.</mark>

In the case of using ReLU on the Fashion MNIST dataset, it's difficult to say exactly what the issue might be without more information about your code and how you're using the ReLU function. However, some potential causes of the segmentation fault error could include:

* Trying to access an invalid memory location when applying the ReLU function.
* Memory allocation issues, such as running out of memory or trying to allocate too much memory.
* Bugs in your code that cause it to access memory in unexpected ways.

To resolve the issue, you can try debugging your code by carefully examining your implementation of the ReLU function and checking for any issues related to **memory allocation** or **uninitialized variables**.

You may also want to try running your code with **smaller subsets of the data** to see if the issue is related to the size of your dataset.

Additionally, using debugging tools such as **gdb** or **valgrind** can help you identify the specific line of code that is causing the segmentation fault error.

> HECC.  I'm doing CIRCLES this time; not FASHION.  POOP!

I just copy/pasted; it should be right.  Especially since I ran it on the GPU and it all worked.

<span style="color:#0000dd;font-size:larger;">Is it possible to get that error simply because I'm using CPU and not GPU?</span>

**It's possible** that using CPU instead of GPU can cause a "segmentation fault" error, but it would be unusual for that to be the sole cause of the error.

In general, using a CPU instead of a GPU can result in slower performance for deep learning tasks, but it should not cause your program to crash or produce a segmentation fault error. However, if you are running out of memory while processing the data, it's possible that using a GPU with more memory could help alleviate the issue.

That being said, it's important to note that a segmentation fault error is typically caused by **issues related to memory allocation and access,** rather than hardware differences. So if you are experiencing a segmentation fault error, it's more likely to be related to **a bug in your code** or an **issue with the dataset** you're using.

<center>To err is to be human<br>
To debug, divine.</center>

<span style="color:#0000dd;font-size:larger;">What does it mean if you get "segmentation fault" on pytorch loss.backward()?</span>

The error typically indicates a **bug in the program's memory management.**

In the context of PyTorch, a segmentation fault during the `loss.backward()` operation could indicate that there is a memory access issue related to the backward pass computation. This could be due to a variety of reasons, such as a bug in the code or running out of memory.

Here are a few steps you could take to try and resolve the issue:

1. **Check for bugs:** Review your code for any potential issues that could be causing the segmentation fault. Make sure that all required tensors have been initialized, and that you're not trying to access tensors that are out of scope.

2. **Reduce the batch size:** If you're running out of memory during the backward pass computation, you could try reducing the batch size. This will reduce the amount of memory required to perform the computation.

3. **Increase memory allocation:** If you're still running out of memory, you could try increasing the memory allocation for your PyTorch session. You can do this by setting the `CUDA_LAUNCH_BLOCKING` environment variable to `1`.

4. **Upgrade PyTorch version:** If you are running an old version of PyTorch, it may be worth upgrading to the latest version as there may be bug fixes related to the backward pass computation.

If none of the above steps resolve the segmentation fault issue, you may need to seek assistance from the **PyTorch community** or consult with an expert in PyTorch programming.

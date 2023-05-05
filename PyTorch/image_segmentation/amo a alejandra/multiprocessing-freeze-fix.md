## Multiprocessing Freeze Fix

<span style="color:#0000dd;">I'm training my U-Net model, I'm looping over the training set.  As soon as I do "for (i, (x, y)) in enumerate(trainLoader):" (where trainLoader is a custom data loader), I get "An attempt has been made to start a new process before the current process has finished its bootstrapping phase..." and then something about "freeze\_support".  What's going on?</span>

When you see all this garbage on the screen, and somewhere it says "freeze support", do this immediately:

<mark>**SET NUM\_WORKERS TO ZERO**</mark>

The error message you're seeing suggests that you're running your code in a multiprocessing environment, such as when using PyTorch's `DataLoader` with multiple workers. 

The issue with starting a new process before the current process has finished its bootstrapping phase arises when you're running on a Windows machine. In Windows, when a new process is started, the operating system needs to create a new Python interpreter instance, which can cause issues with multiprocessing.

<span style="color:red;">Mac is not too happy with it, either.</span>

One solution to this problem is to wrap the main code block of your script with an `if __name__ == '__main__':` statement, and call the `multiprocessing.freeze_support()` function inside it. This tells Python to freeze the current process, allowing it to be spawned into a new process without causing conflicts.

```python
import multiprocessing

def train():
    # Your training code here
    pass

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
```

This should solve the issue you're seeing with `freeze_support`, and allow your script to run without any problems.

<br>

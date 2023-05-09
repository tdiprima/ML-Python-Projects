<span style="color:#00e6c1;font-size:larger;font-weight:bold">I'm using ReLU on the fashion mnist dataset.  But it's saying "31530 segmentation fault".  What's that supposed to mean?</span>

Here.  [StackOver](https://stackoverflow.com/questions/49414841/process-finished-with-exit-code-139-interrupted-by-signal-11-sigsegv#49414907)

"This indicates a bug in your program. In a Python program, this is either a bug in the interpreter or in an extension module being used (and the latter is the most common cause)."

## I'd like to solve the puzzle

These have to be at the end of the script, just before we use 'em.  The End.

```python
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary
```

<br>

Because the guy from StackOverflow was right about the 3rd party crap.

Also...

I reduced code #2 from 1000 to 256 samples, and it worked.

Another thing that can mess it up: IMPORT ORDER MATTERS.

Now it's all good.  With 1000 samples.

Also... remember to put the `params` param; don't just leave it:

```python
# Create an optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
# vs
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

It's important to note that a segmentation fault error is typically caused by **issues related to memory allocation and access,** rather than hardware differences. So if you are experiencing a segmentation fault error, it's more likely to be related to **a bug in your code** or an **issue with the dataset** you're using.

If you're running out of memory during the backward pass computation, you could try reducing the batch size. This will reduce the amount of memory required to perform the computation.

<center>To err is to be human<br>
To debug, divine.</center>

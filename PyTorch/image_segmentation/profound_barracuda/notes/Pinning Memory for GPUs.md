## Pinning Memory for GPUs

There's a thing in pytorch called "pinning memory during data loading".  Why can the gpu access pytorch's container thingy on the cpu fast, but it can't access that same data on that same cpu fast? 🤨

OK, let's imagine your computer is like a busy kitchen. In this kitchen, you have a chef (the GPU) who is super fast at chopping vegetables, stirring pots, and doing all sorts of chef things. Then you have an assistant (the CPU) who brings ingredients to the chef.

Now, the chef has his own special cutting board (GPU memory) where he works really fast, but he can't walk around to get ingredients. He relies on the assistant to bring everything to his special cutting board. The assistant has a different table (CPU memory) where all the ingredients (your data) are initially kept.

So, what does "pinning memory" mean in this situation?

Imagine the assistant has a special tray. When vegetables (your data) are on this special tray, the assistant can quickly slide them over to the chef's special cutting board. If the vegetables are not on this special tray, then the assistant has to spend extra time carefully placing each vegetable on the chef's cutting board one by one.

Pinning memory is like using that special tray. When you pin memory in PyTorch, you're telling the CPU to prepare your data in a way that can be sent over to the GPU super quickly. The CPU and the GPU have different kinds of memory, and data has to be moved carefully from one to the other. Pinning memory helps make this process faster.

So, it's not that the GPU can't access the data on the CPU fast; it's just that the "handover" of this data can be sped up if you prepare it in a special way. This way, your super-fast chef (the GPU) doesn't have to wait too long and can keep on cooking!

## So, like...

The CPU and GPU have separate memory spaces, and we all know how slow copying data over can be.

To improve the performance of data loading, PyTorch provides a feature called "pinned memory". Pinned memory is a special type of memory that is allocated in the CPU memory and can be directly accessed by the GPU without the need for data to be copied.

So that's why we care, and how it works. Now, how do we use it?

To pin the memory during data loading in PyTorch, we can use the `pin_memory` argument when creating a DataLoader object:

```py
from torch.utils.data import DataLoader

# create a dataset
dataset = ...

# create a DataLoader with pinned memory
dataloader = DataLoader(dataset, pin_memory=True, ...)
```

Setting `pin_memory=True` will create a DataLoader that loads data into pinned memory, which can then be directly accessed by the GPU.

<br>

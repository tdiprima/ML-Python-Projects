## Troubleshooting

How do I switch my pytorch program to GPU?

I did: `conda install cudatoolkit`.

But:

```py
device = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = cpu
```

If device is set to `cpu` even though you have installed `cudatoolkit` and `torch.cuda.is_available()` is returning `True`, it is likely that **you have not explicitly instructed PyTorch to use the GPU.**

<span style="color:red;font-weight:bold;">It returns "False".  Scroll down.</span>

## .to(device)

To switch your PyTorch program to use the GPU, you need to move your tensors and model to the device by calling `.to(device)` on each of them.

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Create a tensor on the CPU
x = torch.randn(3, 3)

# Move the tensor to the device
x = x.to(device)

# Create a model
model = torch.nn.Linear(3, 3)

# Move the model to the device
model = model.to(device)
```

In this example, we first check if a GPU is available using `torch.cuda.is_available()`. We then create a tensor on the CPU and move it to the device using `.to(device)`. Similarly, we create a model and move it to the device. Once everything is on the device, you can perform computations on the GPU by calling the appropriate methods on your tensors and models.

If you're using a GPU, you should see a performance improvement over using a CPU. However, keep in mind that not all operations can be accelerated by the GPU, so your mileage may vary.

## torch.cuda.is_available() is returning False

If `torch.cuda.is_available()` is returning False, it means that PyTorch is not able to detect your GPU.

Here are some things you can try to diagnose the issue:

<span style="color:red;font-weight:bold;">This is on "grinder", so...</span>

1. Make sure that your GPU is properly installed and configured. Check that your GPU drivers are up-to-date and that your GPU is properly connected to your computer.

1. Make sure that CUDA is installed on your system. CUDA is a parallel computing platform and programming model that is used by PyTorch to run on the GPU. You can download CUDA from the NVIDIA website.

1. Make sure that you have installed the correct version of PyTorch that supports your version of CUDA. You can check the supported versions of PyTorch and CUDA from the PyTorch website.

1. If you're using a virtual environment, make sure that you have installed PyTorch and CUDA inside the virtual environment.

1. If you're running your code on a remote server, make sure that the server has a GPU and that it's properly configured.

1. Try running `nvidia-smi` to check if your GPU is properly detected by your system. If it's not detected, it could be a hardware issue.

Once you've identified the issue, you can take steps to resolve it. If you're still having trouble, you can try asking for help on the PyTorch forums or Stack Overflow.

# Solved

Well that answers it.

```sh
-bash: nvidia-smi: command not found
```

## GPU stuff

If the `nvidia-smi` command does not work, it could mean that your system is not able to detect your GPU or that there is an issue with your GPU drivers.

Here are some steps you can take to diagnose and resolve the issue:

1. Check that your GPU is properly connected to your computer and that it has power. You may need to re-seat your GPU to ensure that it's properly connected.

1. Check that your GPU drivers are up-to-date. You can download the latest drivers from the NVIDIA website.

1. Check that your GPU is properly detected by your system. You can check this in your system's device manager.

1. Check that your GPU is compatible with your version of CUDA. You can check the supported GPUs for each version of CUDA on the NVIDIA website.

1. Check that your GPU is not being used by another process. Sometimes other processes can lock the GPU and prevent other processes from using it.

If none of these steps resolve the issue, it could be a hardware issue with your GPU. In that case, you may need to replace your GPU or contact your GPU manufacturer for support.

<br>

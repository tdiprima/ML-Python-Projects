## Troubleshooting

<span style="color:#0000dd;">How do I switch my PyTorch program to GPU?</span>

<span style="color:#0000dd;">I did: `conda install cudatoolkit`.</span>

<span style="color:#0000dd;">But:</span>

```py
device = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = cpu
```

If device is set to `cpu` even though you have installed `cudatoolkit` and `torch.cuda.is_available()` is returning `True` <span style="color:red;">(it's not)</span>, it is likely that you have not explicitly instructed PyTorch to use the GPU.

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

<br>

In this example, we first check if a GPU is available using `torch.cuda.is_available()`. We then create a tensor on the CPU and move it to the device using `.to(device)`. Similarly, we create a model and move it to the device. Once everything is on the device, you can perform computations on the GPU by calling the appropriate methods on your tensors and models.

If you're using a GPU, you should see a performance improvement over using a CPU. However, keep in mind that not all operations can be accelerated by the GPU, so your mileage may vary.

## torch.cuda.is_available() is returning False

If `torch.cuda.is_available()` is returning False, it means that PyTorch is not able to detect your GPU.

Here are some things you can try to diagnose the issue:

<span style="color:red;font-weight:bold;">This is on "grinder", so...</span>

1. First of all, make sure that the server has a GPU and that it's properly configured.

    Try running `nvidia-smi` to check.

1. Make sure that CUDA is installed on your system. CUDA is a parallel computing platform and programming model that is used by PyTorch to run on the GPU. You can download CUDA from the NVIDIA website.

1. Check that your GPU drivers are up-to-date and that your GPU is properly connected to your computer.

1. Make sure that you have installed the correct version of PyTorch that supports your version of CUDA. You can check the supported versions of PyTorch and CUDA from the PyTorch website.

1. If you're using a virtual environment, make sure that you have installed PyTorch and CUDA inside the virtual environment.

```sh
# I just ran nvidia-smi and gave up.
-bash: nvidia-smi: command not found
```

## GPU Check Methods

On Red Hat Linux, there are several ways to check if the server has a GPU.

1. Using `lspci` command:

    `lspci` is a command-line utility that lists all the PCI devices connected to your system. Most GPUs are connected through the PCI interface, so you can use this command to check if a GPU is present on your server.

    Open a terminal and run the following command:

    ```sh
    lspci | grep -i vga
    ```

    This command will search for the VGA controller, which usually indicates the presence of a GPU. If a GPU is detected, it will be listed in the output.

2. Using `lshw` command:

    `lshw` is another command-line utility that provides detailed information about the hardware on your system. It is not always installed by default, so you may need to install it using the package manager (`sudo dnf install lshw`).

    Once installed, run the following command:

    ```sh
    sudo lshw -C display
    ```

    This command will show information about all display controllers, which should include any GPUs if they are present.

3. Using NVIDIA System Management Interface (nvidia-smi):

    If your server has an NVIDIA GPU and the NVIDIA drivers are installed, you can use the `nvidia-smi` command to get information about the GPU.

    Simply run:

    ```sh
    nvidia-smi
    ```

    This will provide detailed information about the NVIDIA GPU, including its model, driver version, and utilization.

4. Using `inxi` command:

    `inxi` is a versatile system information script that can provide a wide range of details about your system, including information about GPUs.

    If it's not already installed, you can install it using:

    ```sh
    sudo dnf install inxi
    ```

    Then run:

    ```sh
    inxi -G
    ```

    This will show detailed information about your GPU.

These methods should help you determine whether your server has a GPU and get more information about it if one is present.

<br>

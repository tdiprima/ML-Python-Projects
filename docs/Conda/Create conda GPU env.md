## First of all, check first.

### TensorFlow

```python
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

### PyTorch

```python
import torch

torch.cuda.is_available()
# True

torch.cuda.device_count()
# 1

torch.cuda.current_device()
# 0

torch.cuda.device(0)
# <torch.cuda.device at 0x7efce0b03be0>

torch.cuda.get_device_name(0)
# 'GeForce GTX 950M'
```

## Create conda GPU env

If I've got a python program that I want to run on a GPU, how do I set up a conda environment on that GPU so I can run it?

To set up a conda environment on a GPU, you will need to follow these general steps:

1. **Install** the necessary GPU drivers and libraries on the machine where you want to run your Python program. This will depend on the specific GPU you are using, so refer to the documentation for your GPU for instructions.

1. **Install** CUDA Toolkit and cuDNN, which are essential for running deep learning frameworks on GPUs. These can be downloaded from the NVIDIA website.
    * **CUDA** is a parallel computing platform and programming model developed by NVIDIA for general computing on GPUs.
    * **CuDNN** is a library of primitives for deep neural networks.
    * You can download and install CUDA and CuDNN from the NVIDIA website.

1. **Install** the Anaconda distribution, if you haven't already, which will give you access to the conda package manager.

1. **Create** a new conda environment for your Python program by running the following command in your terminal:

```sh
conda create --name gpu_env
```

Replace "gpu_env" with the name you want to give your environment.

1. **Activate** your new conda environment by running the following command in your terminal:

```sh
conda activate gpu_env
```

1. **Install** the necessary packages and libraries for your Python program using the conda package manager.
    * For example, if you are using TensorFlow and PyTorch, you can install them by running the following commands:

```sh
conda install tensorflow-gpu
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

# conda install pytorch torchvision torchaudio cudatoolkit=<CUDA version>
```

These commands will install TensorFlow with GPU support, and PyTorch with CUDA 11.1 support.

1. **Run** your Python program in your new conda environment. You should see that it is running on the GPU.

```sh
python my_program.py
```

That's it! You now have a conda environment set up on a GPU and can run your Python program with GPU acceleration.

<br>

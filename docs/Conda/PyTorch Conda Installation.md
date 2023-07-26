### Caveat

Sometimes, <mark>**DON'T BOTHER WITH CONDA ENVS. JUST PIP INSTALL.**</mark>

But, if you must, here's how to do it.

Also note that if things go pear-shaped, you may need to uninstall or delete the env and start over.

## Install PyTorch via Anaconda

[pytorch.org](https://pytorch.org/get-started/locally/#mac-anaconda)

```sh
conda install pytorch torchvision -c pytorch
```

### Add conda forge!

```sh
conda config --add channels conda-forge
# May as well add pytorch
conda config --add channels pytorch
```

### What does -c mean?

The flag `-c` is short for `--channel`.

The `-c` flag specifies the package channel from which Conda should install the package. In this case, the package channel is `pytorch`. The PyTorch channel contains the latest versions of PyTorch and related packages.

By specifying `-c pytorch`, Conda will search for the PyTorch package in the PyTorch channel instead of the default channel.

<br>

## How to create Python environment from requirements.txt

Can I set up a Python environment that has a requirements.txt, without installing `virtualenv`?  Can I do it with `conda`?

[Installing requirements.txt in Conda Environments](https://datumorphism.leima.is/til/programming/python/python-anaconda-install-requirements/)

<span style="color:red;">Yes!</span>

```sh
conda create --name myenv python=3.x
```

<br>
Replace `myenv` with the name you want to give to your environment and 3.x with the version of Python you want to use.

Once you have created the environment, you can activate it:

```sh
conda activate myenv
```

<br>
After activating your environment, you can install the packages listed in your requirements.txt file:

```sh
pip install -r requirements.txt
```

<br>
This will install all the packages listed in the requirements.txt file in the activated environment.

Note that if you want to use conda to manage your Python environment, it is recommended to create a new environment for each project you work on to avoid version conflicts between packages.

## Mais oui.

Your shell has not been properly configured to use 'conda activate'.

Solution:

```sh
conda info | grep -i 'base environment'
# Take the answer from that, and go:
source /usr/local/anaconda3/etc/profile.d/conda.sh
```

## Remove

```sh
conda remove -n myenv --all
```

<br>

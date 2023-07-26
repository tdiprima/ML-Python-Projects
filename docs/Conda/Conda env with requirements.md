## How to create Python environment from requirements.txt

<span style="color:#0000dd;">Can I set up a Python environment that has a requirements.txt, without installing "virtualenv"?  Can I do it with "conda"?</span>

[Installing requirements.txt in Conda Environments](https://datumorphism.leima.is/til/programming/python/python-anaconda-install-requirements/)

<span style="color:green;">Yes!</span>

```sh
conda create --name myenv python=3.x
```

<br>

Replace `myenv` with the name you want to give to your environment and 3.x with the version of Python you want to use.

Once you have created the environment, you can activate it.

```sh
conda activate myenv
```

<br>
After activating your environment, you can install the packages listed in your requirements.txt file.

```sh
pip install -r requirements.txt
```

<br>
This will install all the packages listed in the requirements.txt file in the activated environment.

Note that if you want to use `conda` to manage your Python environment, it is **recommended to create a new environment for each project you work on** to avoid version conflicts between packages.

## Shell configuration

Your shell has not been properly configured to use 'conda activate'.

### Solution

```sh
conda info | grep -i 'base environment'
# Take the answer from that, and go:
source /usr/local/anaconda3/etc/profile.d/conda.sh
```

## Conda Remove Environment

```sh
conda remove -n myenv --all
```

<br>

The `conda remove -n myenv --all` command will remove the entire conda environment named "myenv" along with all the packages that have been installed within it.

- `conda remove`: this is the main command to remove packages or environments in conda.

- `-n myenv`: `-n` is an option for specifying the name of the environment. In this case, `myenv` is the name of the environment.

- `--all`: this option tells conda to remove all packages in the environment.

After running this command, the `myenv` environment will no longer exist, and all the resources it was using will be freed. Please be cautious when using this command, as this operation cannot be undone.

<br>

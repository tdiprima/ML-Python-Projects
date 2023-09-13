### OOOOMGGGG

Trying to install specific version of gym, using conda.  Nothing worked.  And then I needed to install atari.  But it seems like I can only do that with pip.

```sh
pip install gym==0.8.2 -e '.[atari]'
```


### gym repo at version 0.17.0 didn't work

I wanted to try to get the other stuff running.  So instead, I created an environment, and...

This didn't work:

```sh
conda install -c conda-forge gym=0.17.0
```

<br>
Nor did this:

```sh
gym==0.17.0
```

## conda - install a specific package version

[I had to do](https://stackoverflow.com/questions/38411942/anaconda-conda-install-a-specific-package-version)

```sh
conda install -y gym"<0.18.0"
```

<br>
And it still wasn't right, but it was close.

Then I had to install atari.  Nothing freakin worked.

```sh
pip install gym"=0.17.0" -e '.[atari]'
```

## Installing an older version with only some environments

[GitHub Gym Issue](https://github.com/openai/gym/issues/1088)

To install in editable mode with the atari dependencies, you can create a Conda environment and install the package using the -e option:

```sh
conda create -n myenv
conda activate myenv
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'
```

<br>
This will clone the gym repository and install it in editable mode with the atari dependencies. You can then activate the environment and use gym in your Python scripts.

<br>

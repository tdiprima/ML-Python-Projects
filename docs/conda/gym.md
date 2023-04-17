### OOOOMGGGG

Trying to install specific version.  "Equals" won't work.  Workaround:

```sh
conda install gym"<0.18.0"
# -y if you're feeling brazen
conda install -y gym"<0.18.0"
```

And it's still not the right version.  But close.

Maybe I gotta go:

```sh
gym[version='=0.17.0']
```

Does not work:

`conda install -c conda-forge gym=0.17.0`

Now I gotta install atari. So it seems like I can only do this with pip.

```sh
pip install gym==0.8.2 -e '.[atari]'
```


## Brilliant

The gym repo at version 0.17.0 didn't work.  I wanted to try to get the other stuff running.  So instead, I created an environment, and...

This didn't work:

`conda install -c conda-forge gym=0.17.0`

Nor did this:

`gym==0.17.0`

[I had to do](https://stackoverflow.com/questions/38411942/anaconda-conda-install-a-specific-package-version):

```sh
conda install -y gym"<0.18.0"
```

And it still wasn't right, but it was close.

Then I had to install atari.  Nothing freakin worked.

```sh
pip install gym"=0.17.0" -e '.[atari]'
```

[Pip](https://github.com/openai/gym/issues/1088).  Wonderful.

### Ask Buddy

What is the conda way of doing: `pip install gym==0.8.2 -e '.[atari]'`?

The equivalent Conda command to install gym version 0.8.2 and its atari dependencies in editable mode is:

```sh
conda install -c conda-forge gym=0.8.2
```

This will install gym version 0.8.2 from the Conda Forge channel.

Buddy's wrong too.

To install in editable mode with the atari dependencies, you can create a Conda environment and install the package using the -e option:

```sh
conda create -n myenv
conda activate myenv
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'
```

This will clone the gym repository and install it in editable mode with the atari dependencies. You can then activate the environment and use gym in your Python scripts.

<br>

```sh
conda activate
```

CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.

If your shell is Bash or a Bourne variant, enable conda for the current user with

```sh
echo ". /cm/shared/apps/anaconda3/current/etc/profile.d/conda.sh" >> ~/.bashrc
```

or, for all users, enable conda with

```sh
sudo ln -s /cm/shared/apps/anaconda3/current/etc/profile.d/conda.sh /etc/profile.d/conda.sh
```

The options above will permanently enable the 'conda' command, but they do NOT
put conda's base (root) environment on PATH.  To do so, run

```sh
conda activate
```

in your terminal, or to put the base environment on PATH permanently, run

```sh
echo "conda activate" >> ~/.bashrc
```

Previous to conda 4.4, the recommended way to activate conda was to modify PATH in
your ~/.bashrc file.  You should manually remove the line that looks like

```sh
export PATH="/cm/shared/apps/anaconda3/current/bin:$PATH"
```

^^^ The above line should NO LONGER be in your `~/.bashrc` file! ^^^


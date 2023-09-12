## Conda Update Loop

<span style="color:#0000dd;">When I run `conda update conda`, all it does is tell me to update it.  Which the command should have done.  So I use the command it gave me `conda update -n base -c defaults conda`, and again, it tells me to update conda.</span>

<span style="color:#0000dd;">I imagine I have to just start over.  `which conda` is $HOME/anaconda3/bin/conda.</span>

It seems like you're **experiencing a loop** where conda asks to be updated even after you run the update command.

This is not a common issue, but there are a few possible solutions you can try:

1. <mark>**Use Miniconda:**</mark> If you're still having issues, you could consider reinstalling Anaconda or switching to Miniconda, which is a smaller distribution that only includes Python and conda, but can still access all the same packages.

2. <mark>**Reset Conda:**</mark> As a last resort, you can completely reset your conda installation by deleting and reinstalling it.

Before doing so, you may want to export your current environment so that you can easily recreate it in the new installation.

```sh
# Export current environment
conda env export > environment.yml
```

<br>

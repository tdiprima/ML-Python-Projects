[GitHub Repo](https://github.com/amueller/introduction_to_ml_with_python)

# Introduction to Machine Learning with Python

This repository holds the code for the forthcoming book "Introduction to Machine
Learning with Python" by [Andreas Mueller](http://amueller.io) and [Sarah Guido](https://twitter.com/sarah_guido).


This repository provides the notebooks from which the book is created, together
with the ``mglearn`` library of helper functions to create figures and
datasets.

All datasets are included in the repository, with the exception of the aclImdb dataset, which you can download from
the page of [Andrew Maas](http://ai.stanford.edu/~amaas/data/sentiment/). See the book for details.

If you get `ImportError: No module named mglearn` you can try to install mglearn into your python environment using
the command `pip install mglearn` in your terminal or `!pip install mglearn` in Jupyter Notebook.

## Setup

### Installing packages with conda:

You can get all packages by running

```sh
conda install numpy scipy scikit-learn matplotlib pandas pillow graphviz python-graphviz
```

For the chapter on text processing you also need to install `nltk` and `spacy`:

```sh
conda install nltk spacy
```

### Installing packages with pip

```sh
pip install numpy scipy scikit-learn matplotlib pandas pillow graphviz
```

You also need to install the graphiz C-library, which is easiest using a package manager.

If you are using OS X and homebrew, you can `brew install graphviz`. If you are on Ubuntu or debian, you can `apt-get install graphviz`.

For the chapter on text processing you also need to install `nltk` and `spacy`:

```sh
pip install nltk spacy
```

### Downloading English language model
For the text processing chapter, you need to download the English language model for spacy using

```sh
python -m spacy download en
```

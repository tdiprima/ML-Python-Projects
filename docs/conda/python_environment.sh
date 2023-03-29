#!/usr/bin/env bash

conda info --envs

conda create --name myenv

#
# To activate this environment, use:
# > conda activate myenv
#
# To deactivate an active environment, use:
# > conda deactivate
#

# It used to be: source activate myenv
conda activate myenv

conda install pymongo shapely scipy opencv pandas matplotlib cython scikit-image

conda install git pip
pip install git+git://github.com/jrosebr1/imutils@master

conda update -n base -c defaults conda

# https://anaconda.org

# Update conda
conda update -n base -c defaults conda

# https://scikit-learn.org/stable/install.html
conda create -n sklearn-env -c conda-forge scikit-learn
conda activate sklearn-env

conda install -c anaconda cudatoolkit

# https://stackoverflow.com/questions/39977808/anaconda-cannot-import-cv2-even-though-opencv-is-installed-how-to-install-open
conda config --add channels menpo
conda install opencv

conda env remove -n name
conda info --envs
conda search openslide

# You can create a copy of the environment by using 
conda env export > environment.yml 
# on all platforms 
# and then create that env with 
conda env create -f environment.yml

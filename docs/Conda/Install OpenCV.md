## Install OpenCV (cv2) with Conda

<span style="color:red;">Answer:</span> I had to create a new env and install opencv first!

```sh
conda create --name tf
conda activate tf
conda install -c conda-forge opencv
# Then:
conda install tensorflow
conda install scikit-learn
```

<br>

There could be a number of reasons why the cv2 installation is failing, such as conflicting dependencies or issues with the package itself. However, there are a few steps you can take to try and resolve the issue:

1. **Update Conda:** Try running `conda update conda` and then retry the installation.

2. <mark>**Create a new environment:**</mark> Sometimes the issue could be caused by conflicts with other packages in your current environment. Try creating a new environment with `conda create --name env_name`, activate it with `conda activate env_name`, and then attempt to install OpenCV.

3. <mark>**Install OpenCV using pip:**</mark> Instead of using Conda, try installing OpenCV using pip. Run `pip install opencv-python`.

<br>

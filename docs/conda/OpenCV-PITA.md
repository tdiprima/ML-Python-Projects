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

`conda update conda` isn't working; all it does is tell me to update it! `conda update -n base -c defaults conda`

I imagine I have to just start over.  `which conda` is `$HOME/anaconda3/bin/conda`.

<br>

There could be a number of reasons why the installation is failing, such as conflicting dependencies or issues with the package itself. However, there are a few steps you can take to try and resolve the issue:

1. **Update Conda:** Try running `conda update conda` and then retry the installation.

2. **Create a new environment:** Sometimes the issue could be caused by conflicts with other packages in your current environment. Try creating a new environment with `conda create --name env_name`, activate it with `conda activate env_name`, and then attempt to install OpenCV.

3. **Install OpenCV using pip:** Instead of using Conda, try installing OpenCV using pip. Run `pip install opencv-python`.

4. **Check system requirements:** Make sure your system meets the requirements for OpenCV. Check the OpenCV documentation for details on system requirements.

5. **Try installing a specific version:** If the latest version of OpenCV isn't installing, try installing an older version with `conda install opencv=x.x.x` (replace x.x.x with the specific version number you want to install).

If none of these steps work, you could try asking for help on the Conda or OpenCV forums or opening an issue on the respective Github repositories.


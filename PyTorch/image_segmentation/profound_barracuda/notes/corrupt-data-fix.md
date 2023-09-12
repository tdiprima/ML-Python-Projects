## CV2 DataLoader Error

TL;DR: The dataset was corrupt.  Like geez, Kaggle... 😒

```sh
find . -iname "._*" -exec rm -- "{}" \; # and that's to show how old it is!
```

I created a pytorch DataLoader `train_loader`.  When I go to train it using `for (i, (x, y)) in enumerate(train_loader):` it works on my machine, but on a server I'm getting an error on it.  But it's a `Type <class 'cv2.error'>` error.  It says "Caught error in DataLoader worker process 0."

The error you are encountering suggests that there might be **an issue with the images** that you are trying to load with **OpenCV** (`cv2`). This could be due to a variety of reasons, such as:

1. The images are corrupted or not in a supported format.
    * Check the file paths
    * Check the image format: `os.path.splitext()`

2. There is not enough memory to load the images.
    * Check the image size
        * Use the `cv2.imread()` function to load a single image and check its size using the `shape` attribute.

3. There is a compatibility issue with the version of OpenCV installed on the server.
    * Check the version of OpenCV: `cv2.__version__`

**Try a smaller subset of the data:**

If the above steps do not work, try using a smaller subset of the data to identify if the issue is with a specific image or a more general problem.

<br>

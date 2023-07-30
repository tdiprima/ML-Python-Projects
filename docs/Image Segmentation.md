## Python Image Segmentation

Using Python and **OpenCV** library to **segment** an **image with dots.** 

* Dot detection <span style="font-size: 20px;">🔵</span>
* Circles the dots <span style="font-size: 25px;">🫥</span>

[image-segmentation.py](../bear_training/image-segmentation.py)

### Explanation of code:

1. We **load** the image using the `cv2.imread()` function. <span style="font-size: 27px;">📖</span>

2. We convert the image to **grayscale** using the `cv2.cvtColor()` function. <span style="font-size: 27px;">🐺

3. We apply a threshold to **create a binary** image using the `cv2.threshold()` function. <span style="font-size: 27px;">🐼
    * The `cv2.THRESH_BINARY_INV` flag **inverts** the binary image <span style="font-size: 27px;">🙃
    * The `cv2.THRESH_OTSU` flag automatically **calculates** the threshold value.
        * In computer vision and image processing, Otsu's method, named after Nobuyuki Otsu (大津展之, Ōtsu Nobuyuki), is used to perform automatic image thresholding. <span style="font-size: 27px;">🇯🇵🏯

4. We find **contours** in the binary image using the `cv2.findContours()` function. <span style="color: blue; font-size: 30px;">∮
    * **RETR\_EXTERNAL** – retrieves only the extreme outer contours.
    * **CHAIN\_APPROX\_SIMPLE** – returns only the endpoints that are necessary for drawing the contour line.

5. For each contour, we:
    * Find the **minimum enclosing circle** using the `cv2.minEnclosingCircle()` function <span style="font-size: 27px;">🟢
    * **Draw a circle around the contour** using the `cv2.circle()` function.

6. **Display** the resulting image using the `cv2.imshow()` function <span style="font-size: 27px;">🎪

7. **Wait** for a **key press** using the `cv2.waitKey()` function <span style="font-size: 27px;">✋🏼

**Note:** The quality of the segmentation will depend on the quality of the input image and the parameters used in the code.

You may need to adjust the threshold value and/or the contour detection parameters to obtain the desired results.

## OpenCV's Name 🖥️ 👀

OpenCV stands for "Open Source Computer Vision".

<br>

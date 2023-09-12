## Python Image Segmentation

Using Python and **OpenCV** (Open Source Computer Vision) library to **segment** an **image with dots.** 

* Dot detection <span style="font-size: 20px;">🟢</span>
* Circles the dots <span style="font-size: 25px;">🫥</span>

[image-segmentation.py](../bear_training/image-segmentation.py)

## Explanation of code:

1. Load the image using the `cv2.imread()` function.

2. Convert the image to grayscale using the `cv2.cvtColor()` function.

3. Apply a threshold to create a binary image using the `cv2.threshold()` function.
    * The `cv2.THRESH_BINARY_INV` flag inverts the binary image
    * The `cv2.THRESH_OTSU` flag automatically calculates the threshold value.
        * In computer vision and image processing, Otsu's method, named after Nobuyuki Otsu (大津展之, Ōtsu Nobuyuki), is used to perform automatic image thresholding.

4. Find contours in the binary image using the `cv2.findContours()` function. <span style="color: blue; font-size: 30px;">∮</span>
    * RETR\_EXTERNAL – retrieves only the extreme outer contours.
    * CHAIN\_APPROX\_SIMPLE – returns only the endpoints that are necessary for drawing the contour line.

5. For each contour:
    * Find the minimum enclosing circle using the `cv2.minEnclosingCircle()` function
    * Draw a circle around the contour using the `cv2.circle()` function.

6. Display the resulting image using the `cv2.imshow()` function

7. Wait for a key press using the `cv2.waitKey()` function

==**Note:**== The quality of the segmentation will depend on the quality of the input image and the parameters used in the code.

You may need to adjust the threshold value and/or the contour detection parameters to obtain the desired results.

<br>

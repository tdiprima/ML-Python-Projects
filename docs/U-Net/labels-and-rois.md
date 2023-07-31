<div align="center" style="color:#880000;">A review of what I did for MICCAI 2015</div>

## Making labels for a WSI

To create labels for a slide image, you need to manually annotate the regions of interest (ROIs) in the slide image that correspond to the classes you want to segment. This process is commonly referred to as **ground truth annotation** or image labeling. Here are the steps you can follow to label a slide image:

1. Choose the **tool** for image labeling: There are various tools available for annotating images such as VGG Image Annotator (VIA), Labelbox, and RectLabel.

    iPhotoDraw was a super-simple user-friendly one, to annotate the patches.

    Aperio ImageScope - Pathology Slide Viewing Software.

3. Define the **classes** you want to segment: Decide on the classes you want to segment in your slide image. For example, in medical imaging, you might want to segment the tumor region, healthy tissue, and background.

4. Load the slide image in the labeling tool: Load the slide image in the labeling tool you have chosen.

5. Manually annotate the ROIs: Use the tools provided by the labeling tool to draw the ROIs that correspond to the classes you want to segment. Make sure that the ROIs are accurately drawn, and there is no overlap between different ROIs.

6. Save the labels: Once you have annotated all the ROIs in the slide image, save the labels in a format that can be read by your deep learning framework. For PyTorch, you can save the labels as PNG images, where each pixel corresponds to a class.

7. Repeat the process for all slide images: Repeat the above process for all the slide images you have, making sure to annotate the ROIs accurately and consistently.

Labeling a large number of slide images can be a time-consuming process, and it requires expertise in identifying the ROIs accurately. Therefore, it is essential to allocate enough time and resources for the labeling process to ensure high-quality ground truth annotations.

## Extracting ROIs that correspond to the labels

Actually, I divvied-up the slide first, using `vips` or `OpenSlide`.  (Starting from the center of the image, because theoretically, there are more cells there.  Ya don't wanna do random coordinates and extracting only slide and no tissue!)

```sh
# openslide-write-png [OPTION...] slide x y level width height output.png
openslide-write-png 'TCGA-CV-6945-01Z-00-DX1.F2533BF5-3C26-43BB-A5D1-5CEA4B065D6C.svs' 117830 22430 0 512 512 'TCGA-CV-6945-01Z-00-DX1_117830_22430_512_512.png'
```

Once you have annotated the regions of interest (ROIs) in your slide images and saved them as labels, you can use these labels to extract the corresponding ROIs. Here are the steps you can follow to extract ROIs from labeled slide images:

1. Load the slide image and the label: Load the slide image and the corresponding label into your Python environment using a library such as **OpenCV or Pillow** (I used PIL).

2. Convert the label to a binary mask: Convert the label to a **binary mask**, where each pixel in the mask corresponds to a class. For example, if you have two classes, "tumor" and "healthy tissue", you can set the pixels corresponding to "tumor" to 1 and the pixels corresponding to "healthy tissue" to 0.

3. Apply the binary mask to the slide image: Apply the binary mask to the slide image to extract the ROIs. You can do this by multiplying the slide image by the binary mask, where the pixels corresponding to the ROI will be preserved, and the pixels corresponding to the background will be set to 0.

4. Crop the ROI: Crop the extracted ROI from the slide image using the coordinates of the ROI. You can use the `cv2.boundingRect()` function in OpenCV to get the **coordinates** of the bounding box around the ROI.

5. Repeat the process for all slide images: Repeat the above process for all the slide images and corresponding labels in your dataset to extract the ROIs.

Here is some sample code that shows how to extract ROIs from a labeled slide image using OpenCV:

```python
import cv2

# Load the slide image and the corresponding label
slide_image = cv2.imread('path/to/slide/image')
label = cv2.imread('path/to/label')

# Convert the label to a binary mask
binary_mask = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
binary_mask[binary_mask != 255] = 0
binary_mask[binary_mask == 255] = 1

# Apply the binary mask to the slide image
extracted_roi = cv2.multiply(slide_image, binary_mask)

# Get the coordinates of the bounding box around the ROI
x, y, w, h = cv2.boundingRect(binary_mask)

# Crop the ROI
cropped_roi = extracted_roi[y:y + h, x:x + w]
```

<br>

Note that the above code assumes that the label is saved as a grayscale image, where the ROI corresponds to the white pixels (pixel value of 255). You may need to adjust the code depending on the format and conventions used for your labels.

<br>

## Resize to nearest interpolation

Resizing an image to the nearest interpolation means that each pixel in the new image is assigned the value of the nearest pixel in the original image. This is the simplest type of interpolation method used in image resizing.

For example, let's say you have a **4x4** pixel image that you want to resize to a **2x2** pixel image. When you resize the image using nearest interpolation, each pixel in the new image is assigned the value of the nearest pixel in the original image. In this case, the new image would be:

```
Original Image        Resized Image
[1 2 3 4]             [1 3]
[5 6 7 8]   -->       [5 7]
[9 10 11 12]          [9 11]
[13 14 15 16]
```

<br>

The pixel values of the new image are assigned based on the nearest pixel in the original image. The pixel (1,1) in the new image is assigned the value of pixel (1,1) in the original image, the pixel (1,2) in the new image is assigned the value of pixel (1,3) in the original image, and so on.

## Code example

You can use the `PIL` (Python Imaging Library) library in Python to resize an image to the nearest interpolation.

```python
from PIL import Image

# Open the original image
image = Image.open("original_image.jpg")

# Resize the image to the nearest interpolation
new_size = (300, 200)
# resized_image = image.resize(new_size, resample=Image.NEAREST)
resized_image = image.resize(new_size, resample=Image.Resampling.NEAREST)


# Save the resized image
resized_image.save("resized_image.jpg")
```

<br>

We use the `resize()` method to resize the image to the nearest interpolation using ~~Image.NEAREST~~ <span style="color:#0000dd;">Image.Resampling.NEAREST</span> as the **resampling filter.**

## THREE.NearestFilter

It's like what three.js does in THREE.NearestFilter.

Yes, that's correct! `THREE.NearestFilter` is used as a **texture filtering** option to specify that the texture should be resized using the nearest interpolation method. 

When you create a texture in Three.js, you can specify a filtering option to use for resizing the texture. 

Create a new texture with `THREE.NearestFilter` as the **filtering option:**

```javascript
var texture = new THREE.TextureLoader().load( 'texture.jpg' );
texture.magFilter = THREE.NearestFilter;
texture.minFilter = THREE.NearestFilter;
```

<br>

We set the `magFilter` and `minFilter` properties of the texture to `THREE.NearestFilter`, which tells Three.js to **use nearest interpolation** when resizing the texture. 

This can be useful in certain cases where you want to **preserve** the pixelated or blocky appearance of the original texture, rather than smoothing it out when resizing.

<br>

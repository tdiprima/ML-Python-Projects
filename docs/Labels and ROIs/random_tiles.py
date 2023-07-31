#!/usr/bin/python
# RANDOM SCENE GENERATOR
__author__ = 'tdiprima'

import random

from gi.repository import Vips

# Load the input file.
# TCGA-NC-A5HG-01Z-00-DX1 is LUSC
# TCGA-49-AARN-01Z-00-DX1 is LUAD
slideName = "TCGA-02-0001-01Z-00-DX1"
im = Vips.Image.new_from_file("/Users/tdiprima/Downloads/" + slideName + ".svs")

for x in range(0, 4):
    # extract_area(left, top, width, height)
    width = im.width
    height = im.height

    print("width: " + str(width))
    print("height: " + str(height))

    left = random.randint(height, width)
    top = random.randint(height, width)

    print("left: " + str(left))
    print("top: " + str(top))

    size = 500

    im1 = im.extract_area(left, top, size, size)

    im1.write_to_file(
        "/Users/tdiprima/Downloads/" + slideName + "_" + str(left) + "_" + str(top) + "_" + str(size) + "_" + str(
            size) + ".png")

print('Done.')

# im = im.extract_area(100, 100, im.width - 200, im.height - 200)
# im = im.similarity(scale=0.9)
# mask = Vips.Image.new_from_array([[-1, -1, -1],
#                                   [-1, 16, -1],
#                                   [-1, -1, -1]], scale=8)
# im = im.conv(mask)

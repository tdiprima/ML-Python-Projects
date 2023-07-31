#!/usr/bin/env python
import getopt
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


def print_usage():
    print('region_parse_multi_region.py -i <image file> -x <region file> -m <output mask> -o <text mask>')


try:
    opts, args = getopt.getopt(sys.argv[1:], "i:x:m:o:")
except getopt.GetoptError:
    print_usage()
    sys.exit(2)

if not opts:
    print_usage()
    sys.exit(2)

ifile = ''
xfile = ''
mfile = ''
ofile = ''
for opt, arg in opts:
    if opt in '-i':
        ifile = arg
    elif opt in '-x':
        xfile = arg
    elif opt in '-m':
        mfile = arg
    elif opt in '-o':
        ofile = arg
    else:
        print_usage()
        sys.exit(2)

if ifile == '' or xfile == '' or mfile == '' or ofile == '':
    print_usage()
    sys.exit(2)

in_img = Image.open(ifile)
(img_xdim, img_ydim) = in_img.size
print(img_xdim, img_ydim)

tumor_region = 0
background_region = 0
necrosis_region = 1

regionid = necrosis_region
labelid = tumor_region
mask_arry = np.zeros((img_xdim, img_ydim), dtype=int)
for j in range(0, img_xdim):
    for i in range(0, img_ydim):
        mask_arry[j, i] = labelid

binary_img = Image.new("RGB", (img_xdim, img_ydim), (0, 0, 255))
pixels = binary_img.load()

# Check if the manual markup file exists.
# If it doesn't, the entire image is TUMOR/NON-NECROSIS
if not os.path.isfile(xfile):
    print('manual markup file does not exist')
    labelid = tumor_region
    for j in range(0, img_xdim):
        for i in range(0, img_ydim):
            mask_arry[j, i] = labelid
    print('ALL TUMOR: Writing textual mask file: ', ofile)
    f = open(ofile, 'w')
    f.write(str(img_xdim))
    f.write(' ')
    f.write(str(img_ydim))
    f.write('\n')
    for j in range(0, img_ydim):
        for i in range(0, img_xdim):
            f.write(str(mask_arry[i, j]))
            f.write('\n')
            if mask_arry[i, j] == tumor_region:
                pixels[i, j] = 255, 255, 255
    f.close()

    print('ALL TUMOR: Writing binary mask file:  ', mfile)
    binary_img.save(mfile)
    sys.exit(1)

# Manual markup file exists.
print('manual markup file exists')
tree = ET.parse(xfile)
root = tree.getroot()
# Unsigned integer.
counter = int(0)

for child in root.iter('Shape'):
    counter += 1
    print('Shape ', counter)

    # Entire image is necrotic region
    if child.get('Type') == 'Rectangle':
        labelid = necrosis_region
        for j in range(0, img_xdim):
            for i in range(0, img_ydim):
                mask_arry[j, i] = labelid
        print('RECTANGLE: Writing textual mask file: ', ofile)
        f = open(ofile, 'w')
        f.write(str(img_xdim))
        f.write(' ')
        f.write(str(img_ydim))
        f.write('\n')
        for j in range(0, img_ydim):
            for i in range(0, img_xdim):
                f.write(str(mask_arry[i, j]))
                f.write('\n')
                if mask_arry[i, j] >= necrosis_region:
                    pixels[i, j] = 0, 0, 0
        f.close()

        print('RECTANGLE: Writing binary mask file:  ', mfile)
        binary_img.save(mfile)
        sys.exit(1)
    else:
        polygon = []
        minx = 10000000
        miny = 10000000
        maxx = 0
        maxy = 0

        for mytext in child.iter('Text'):
            if mytext.text == 'BACKGROUND':
                labelid = background_region
                print('Processing BACKGROUND. Label: ', labelid)
            elif mytext.text == 'TUMOR':
                labelid = tumor_region
                print('Processing TUMOR TISSUE. Label: ', labelid)
            elif mytext.text == 'NECROSIS':
                labelid = necrosis_region
                print('Processing NECROSIS Region. Label: ', labelid)
            else:
                labelid = necrosis_region
                print('Processing NECROSIS Region. Label: ', labelid)

        for points in child.iter('Point'):
            px = points.attrib.get('X')
            py = points.attrib.get('Y')
            ipx = int(float(px) + 0.5)
            ipy = int(float(py) + 0.5)
            if ipx < 0:
                ipx = 0
            if ipx >= img_xdim:
                ipx = img_xdim - 1
            if ipy < 0:
                ipy = 0
            if ipy >= img_ydim:
                ipy = img_ydim - 1
            if minx > ipx:
                minx = ipx
            if maxx < ipx:
                maxx = ipx
            if miny > ipy:
                miny = ipy
            if maxy < ipy:
                maxy = ipy
            if minx <= 0:
                minx = 1
            if miny <= 0:
                miny = 1
            if maxx >= img_xdim:
                maxx = img_xdim - 1
            if maxy >= img_ydim:
                maxy = img_ydim - 1
            polygon += ipx, ipy

        temp_img = Image.new('L', (img_xdim, img_ydim), 0)
        ImageDraw.Draw(temp_img).polygon(polygon, outline=1, fill=1)
        temp_mask = np.array(temp_img)
        for j in range(minx - 1, maxx + 1):
            for i in range(miny - 1, maxy + 1):
                if temp_mask[i, j] == 1:
                    print('mask_arry[j, i] = ', str(counter))
                    mask_arry[j, i] = str(counter)
                    # mask_arry[j, i] = labelid

print('Writing textual mask file: ', ofile)
f = open(ofile, 'w')
f.write(str(img_xdim))
f.write(' ')
f.write(str(img_ydim))
f.write('\n')
for j in range(0, img_ydim):
    for i in range(0, img_xdim):
        f.write(str(mask_arry[i, j]))
        f.write('\n')
        if mask_arry[i, j] == background_region:
            pixels[i, j] = mask_arry[i, j]
            # pixels[i, j] = 0, 255, 255
            # [is cyan]
        if mask_arry[i, j] == tumor_region:
            pixels[i, j] = mask_arry[i, j]
            # pixels[i, j] = 255, 255, 255
            # [is white]
        if mask_arry[i, j] >= necrosis_region:
            pixels[i, j] = mask_arry[i, j]
            # pixels[i, j] = 0, 0, 0
            # [is black]
f.close()
print('Writing binary mask file:  ', mfile)
binary_img.save(mfile)

# If you got this far, congratulations.
# Successful termination.
sys.exit(0)

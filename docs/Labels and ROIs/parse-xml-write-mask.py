#!/usr/bin/env python
# RUN AFTER PATCH-TO-MASK
# Adapted from region_parse_multi_region.py
# Parse XML file to get the polygons, and write to mask file.
# Set option to write debug stuff (TRUE/FALSE).
# If false, then we are writing clean production text containing
# file size and tuples.
__author__ = 'tdiprima'

import getopt
import sys
import xml.etree.ElementTree as eT

import numpy as np
from PIL import Image, ImageDraw

write_debug = False


def proc_pix(f, x, y, arr, pix):
    for jj in range(0, y):
        for ii in range(0, x):
            f.write(str(mask_arry[ii, jj]))
            f.write('\n')
            pix[ii, jj] = arr[ii, jj]
    f.close()
    return pix


def proc_pix_debug(cnt, f, x, y, arr, pix):
    f.write('Shape count: ' + str(cnt))
    f.write('\n')
    for jj in range(0, y):
        for ii in range(0, x):
            pix[ii, jj] = arr[ii, jj]
            tup = pix[ii, jj]
            f.write(str(tup))
            if tup[1] > 0:
                ma = arr[ii, jj]
                f.write(' = ' + str(ma) + '-255 = ' + str(ma - 255 - tup[1]) + '+' + str(tup[1]))
            if tup[2] > 0:
                f.write('####################')
            f.write('\n')

    f.close()
    return pix


def proc_pix_prod(f, x, y, arr, pix):
    for jj in range(0, y):
        for ii in range(0, x):
            pix[ii, jj] = arr[ii, jj]
            tup = pix[ii, jj]
            f.write(str(tup))
            f.write('\n')

    f.close()
    return pix


def print_usage():
    print('Usage:\nxml-to-mask.py -i <image file> -x <xml file> -m <output mask> -o <text mask>')


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

# ifile = 'input/TCGA-05-4244-01Z-00-DX1_19500_19500_500_500_LUAD.png'
# xfile = 'input/TCGA-05-4244-01Z-00-DX1_19500_19500_500_500_LUAD_data.xml'
# mfile = 'output/TCGA-05-4244-01Z-00-DX1_19500_19500_500_500_LUAD_mask.png'
# ofile = 'output/TCGA-05-4244-01Z-00-DX1_19500_19500_500_500_LUAD_text.txt'

in_img = Image.open(ifile)
(img_xdim, img_ydim) = in_img.size
print("img_xdim %d, img_ydim %d" % (img_xdim, img_ydim))

# Set array to zeros
mask_arry = np.zeros((img_xdim, img_ydim), dtype=int)

binary_img = Image.new("RGB", (img_xdim, img_ydim), (0, 0, 255))
pixels = binary_img.load()

# Manual markup file exists.
# print 'manual markup file exists'
tree = eT.parse(xfile)
root = tree.getroot()
counter = int(0)

for child in root.iter('Shape'):
    counter += 1
    # print 'Shape ', counter

    polygon = []
    minx = 10000000
    miny = 10000000
    maxx = 0
    maxy = 0

    # Normalize the polygon boundary
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

    # Making a binary image
    temp_img = Image.new('L', (img_xdim, img_ydim), 0)

    # Draw the polygon on our mask array
    ImageDraw.Draw(temp_img).polygon(polygon, outline=1, fill=1)
    temp_mask = np.array(temp_img)
    for j in range(minx - 1, maxx + 1):
        for i in range(miny - 1, maxy + 1):
            if temp_mask[i, j] == 1:
                mask_arry[j, i] = str(counter)
                # mask_arry[j, i] = labelid

# print 'Writing textual mask file: ', ofile

ff = open(ofile, 'w')
ff.write(str(img_xdim))
ff.write(' ')
ff.write(str(img_ydim))
ff.write('\n')

pixels = proc_pix(ff, img_xdim, img_ydim, mask_arry, pixels)

# if write_debug:
#     pixels = proc_pix_debug(counter, ff, img_xdim, img_ydim, mask_arry, pixels)
# else:
#     pixels = proc_pix_prod(ff, img_xdim, img_ydim, mask_arry, pixels)
# print('Writing binary mask file:', mfile)

binary_img.save(mfile)

sys.exit(1)

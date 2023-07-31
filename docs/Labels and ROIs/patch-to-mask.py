#!/usr/bin/env python
# RUN THIS FIRST
# Main file for converting WSI patches to masks.
# Set option to deidentify (True/False).
# Set option to print TIFF (if False, then PNG).
# This file rolls through the file directory containing the patches.
# parse-xml-write-mask.py does the actual processing.
__author__ = 'tdiprima'

import os

deidentify = True
prt_tiff = False


def de_id(thing, cnt):
    if cnt < 10:
        thing += '0'
    thing += str(cnt)
    return thing


rootDir = 'input'
myPng = ''
myXml = ''

# parse_region.py -i <image file> -x <region file> -m <output mask> -o <text mask>
count = int(0)
# I'm exploiting the way the directory is read sequentially.
# First you will find a png
# Then you will find an xml
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:

        if fname.endswith('png'):
            myPng = fname

        if fname.endswith('xml'):
            count += 1
            myXml = fname

            if deidentify:
                myMsk = 'image'
                myTxt = 'image'
                myMsk = de_id(myMsk, count)

                if prt_tiff:
                    myMsk += '.tiff'
                else:
                    myMsk += '.png'

                myTxt = de_id(myTxt, count)
                myTxt += '_mask.txt'
            else:
                l = len(myPng)
                stub = myPng[0:l - 4]
                if prt_tiff:
                    myMsk = stub + '_mask.tiff'
                else:
                    myMsk = stub + '_mask.png'

                myTxt = stub + '_text.txt'

            myCmd = 'python Python-2.py -i input/' + myPng + ' -x input/' + myXml + ' -m output/' + myMsk + ' -o output/' + myTxt

            print(myCmd)
            os.system(myCmd)

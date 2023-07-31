#!/usr/bin/env python
# Python-1a.py
# Deidentify the handmade images with the circles on them.
__author__ = 'tdiprima'

import os
import sys

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

count = int(0)

try:
    # I'm exploiting the way the directory is read sequentially.
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:

            if fname.endswith('_2.png'):
                count += 1
                myPng = fname

                if deidentify:
                    myMsk = 'image'
                    myMsk = de_id(myMsk, count)

                    if prt_tiff:
                        myMsk += '_2.tiff'
                    else:
                        myMsk += '_2.png'

                else:
                    l = len(myPng)
                    stub = myPng[0:l - 4]
                    if prt_tiff:
                        myMsk = stub + '_2.tiff'
                    else:
                        myMsk = stub + '_2.png'

                print(fname, ' ', myMsk)
                os.rename('input/' + fname, 'output/' + myMsk)
            else:
                print('file does not end in _2.png', fname)
except Exception as e:
    # print("Unexpected error:", sys.exc_info()[0])
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("\nType", exc_type)
    print("\nErr:", exc_obj)
    print("\nLine:", exc_tb.tb_lineno)
    sys.exit(1)

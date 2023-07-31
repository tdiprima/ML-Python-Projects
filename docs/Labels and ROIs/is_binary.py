#!/usr/bin/env python
# Check to see if auto-segmented masks are true binary.
__author__ = 'tdiprima'

import os

from scipy.misc import imread

# rootDir = 'input'
rootDir = '.'

count = int(0)
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        count += 1
        # print('Reading image ' + str(count) + "...")
        print('\nReading image ' + fname + "...")

        img = imread('./' + fname)
        li = img.flatten()
        print('Checking for pixel values greater than 1...')
        for s in li:
            # Check its value
            if s > 1:
                print(s)
                break

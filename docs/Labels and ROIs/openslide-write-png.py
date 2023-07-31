#!/usr/bin/env python
"""
openslide-write-png.py: Extract annotated regions from WSI.
"""
__author__ = 'tdiprima'

import os
import xml.etree.ElementTree as ET

path = "/Users/tdiprima/Downloads"
os.chdir(path)

for name in os.listdir(path):
    if name.endswith(".xml"):
        print(name)
        tree = ET.parse(name)
        root = tree.getroot()

        # Depending on Python version, you will use .iter() or .getiterator().
        # for foo in root.iter("Region"):
        for foo in root.getiterator("Region"):

            lox = 0
            loy = 0
            hix = 0
            hiy = 0

            xs = list()
            ys = list()

            # for vertex in foo.iter('Vertex'):
            for vertex in foo.getiterator('Vertex'):
                # <Vertex X="59985" Y="34177"/>
                m_dict = vertex.attrib
                xs.append(m_dict['X'])
                ys.append(m_dict['Y'])

            lox = min(xs)
            hix = max(xs)
            loy = min(ys)
            hiy = max(ys)

            lox = round(float(lox))
            hix = round(float(hix))
            loy = round(float(loy))
            hiy = round(float(hiy))

            w = hix - lox
            h = hiy - loy

            # '{0:g}'.format(float(21))
            fname = os.path.splitext(name)[0] + "_" + str(abs(lox)) + "_" + str(abs(loy)) + "_" + str(
                abs(w)) + "_" + str(abs(h)) + ".png"

            cmd = "openslide-write-png " + os.path.splitext(name)[0] + ".svs " + str(abs(lox)) + " " + str(
                abs(loy)) + " " + str(0) + " " + str(abs(w)) + " " + str(abs(h)) + " " + path + "/output/" + fname

            print(cmd)
            os.system(cmd)

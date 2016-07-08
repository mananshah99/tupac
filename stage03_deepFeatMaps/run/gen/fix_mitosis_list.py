import os, sys

f = open('../mitosis.lst')
out = open('../mitosis-fixed.lst')

import cv2
import warnings
warnings.simplefilter('error')
for line in f:
    try:
        image = cv2.imread('../' + line.strip('\n'))
    except Exception as e:
        print e
        print line
        continue

    print 'OK: \t', line

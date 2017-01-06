import os
import sys
import subprocess
import cv2
import numpy as np

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
import skimage.io as skio
from skimage.transform import resize
def extract_features(heatmap_mitoses, heatmap_roi, tv, ratio=4.0, dscale=0.0001):
    ALL_BLACK = False
    FEA_NUM = 3
    #print "\t Loading:", heatmap_mitoses
    #print "\t Loading:", heatmap_roi

    try:
        im = skio.imread(heatmap_mitoses, as_grey=True)
        im2 = skio.imread(heatmap_roi, as_grey=True)

        h, w = im.shape
        nh = int(h / ratio)
        nw = int(w / ratio)
        im = resize(im, (nh, nw)) # img_as_float()
        im2 = resize(im2, (nh, nw))
        print "im.shape", im.shape
        print "im2.shape", im2.shape
    except Exception as e:
        print "ERROR"
        ALL_BLACK = True

    vector = []
    for threshold_decimal in [0.5, 0.6, 0.7, 0.8, 0.9]: #based on mitko's threhsolds #np.arange(0.1, 1, 0.1): #much slower with larger images
        print "@@@@@", threshold_decimal
        individual_vector = []

        if ALL_BLACK:
            individual_vector.extend([0]*FEA_NUM)
            vector.extend(individual_vector)
            continue

        bw = im > threshold_decimal
        bw2 = im2 > tv
        # label image regions
        label_image = label(bw)
        potential_mitoses = regionprops(label_image, intensity_image=im)

        mitoses_area = []
        mitoses_eccentricity = []
        for x in potential_mitoses:
            mitoses_area.append(x.area)
            mitoses_eccentricity.append(x.eccentricity)

        if len(mitoses_area) == 0:
            mitoses_area.append(0)
            mitoses_eccentricity.append(0)

        TUMOR_SIZE = np.sum(bw2) * dscale

        num_mitoses = float(len(potential_mitoses)) / TUMOR_SIZE
        #mitoses.append(num_mitoses)
        print "@@@@", len(potential_mitoses), num_mitoses, TUMOR_SIZE
        try:
            individual_vector.append(num_mitoses)
            individual_vector.append(np.average(mitoses_area))
            individual_vector.append(np.average(mitoses_eccentricity))
        except Exception as e: #0 mitoses
            print e
            print e.args
            individual_vector.extend([0]*FEA_NUM)

        # vector is extended at each threshold
        vector.extend(individual_vector)
    return vector

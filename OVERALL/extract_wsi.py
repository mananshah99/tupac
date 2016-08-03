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
from skimage.io import *

def extract_features(heatmap):
    ALL_BLACK = False
    try:
        im = imread(heatmap, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
#        print e
        ALL_BLACK = True

    vector = []
    for threshold_decimal in [0.1, 0.2, 0.3, 0.4, 0.5, 0.67, 0.83, 0.85, 0.91, 0.95, 0.97]: #based on mitko's threhsolds #np.arange(0.1, 1, 0.1): #much slower with larger images

        MANUAL_THRESHOLD = int(255 * threshold_decimal)
        individual_vector = [] 

        if ALL_BLACK:
            individual_vector.extend([0]*31)
            vector.extend(individual_vector)
            continue

        mitoses = []
        mitoses_area = []
        mitoses_eccentricity = []
        mitoses_max_i = []
        mitoses_min_i = []
        mitoses_mean_i = []
        mitoses_solidity = []

        thresh = MANUAL_THRESHOLD
            
        _, bw = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)

        # label image regions
        label_image = label(bw)

        A = 0
        potential_mitoses = regionprops(label_image, intensity_image=im)

        for x in potential_mitoses:
            mitoses_area.append(x.area)
            mitoses_eccentricity.append(x.eccentricity)
            mitoses_max_i.append(x.max_intensity)
            mitoses_min_i.append(x.min_intensity)
            mitoses_mean_i.append(x.mean_intensity)
            mitoses_solidity.append(x.solidity)

        if len(mitoses_area) == 0:
            mitoses_area.append(0)
            mitoses_eccentricity.append(0)
            mitoses_max_i.append(0)
            mitoses_min_i.append(0)
            mitoses_mean_i.append(0)
            mitoses_solidity.append(0)

        num_mitoses = len(potential_mitoses) 
        mitoses.append(num_mitoses)

        try:
            mitoses_area = sorted(mitoses_area)
            if len(mitoses_area[int(0.2 * len(mitoses_area)) : int(0.8 * len(mitoses_area))]) > 1:
                mitoses_area = mitoses_area[int(0.2 * len(mitoses_area)) : int(0.8 * len(mitoses_area))] 
            
            individual_vector.append(num_mitoses)

            individual_vector.extend([np.median(mitoses_area), 
                                      np.average(mitoses_area), 
                                      np.amin(mitoses_area), 
                                      np.amax(mitoses_area), 
                                      np.std(mitoses_area)])

            individual_vector.extend([np.median(mitoses_eccentricity),
                                      np.average(mitoses_eccentricity),
                                      np.amin(mitoses_eccentricity),
                                      np.amax(mitoses_eccentricity),
                                      np.std(mitoses_eccentricity)])

            individual_vector.extend([np.median(mitoses_min_i),
                                      np.average(mitoses_min_i),
                                      np.amin(mitoses_min_i),
                                      np.amax(mitoses_min_i),
                                      np.std(mitoses_min_i)])

            individual_vector.extend([np.median(mitoses_max_i),
                                      np.average(mitoses_max_i),
                                      np.amin(mitoses_max_i),
                                      np.amax(mitoses_max_i),
                                      np.std(mitoses_max_i)])

            individual_vector.extend([np.median(mitoses_mean_i),
                                      np.average(mitoses_mean_i),
                                      np.amin(mitoses_mean_i),
                                      np.amax(mitoses_mean_i),
                                      np.std(mitoses_mean_i)])

            individual_vector.extend([np.median(mitoses_solidity),
                                      np.average(mitoses_solidity),
                                      np.amin(mitoses_solidity),
                                      np.amax(mitoses_solidity),
                                      np.std(mitoses_solidity)])

        except Exception as e: #0 mitoses
            print e
            print e.args
            individual_vector.extend([0]*31)

        # vector is extended at each threshold 
        vector.extend(individual_vector)
    return vector 

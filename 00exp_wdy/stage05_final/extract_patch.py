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

def getXY(name):
    if '(' in name:
        # old format; uses (
        tmp = name.split('(')[1].split(')')[0].split(',')
        h1_level0 = int(tmp[0])
        w1_level0 = int(tmp[1])
    else:
        # new format; uses x__y__
        tmp = name.split('x')[1].split('_')
        h1_level0 = int(tmp[0])
        w1_level0 = int(tmp[1][1:-8]) #.png.png mistake

            # X         Y
    return (w1_level0, h1_level0)
    
def do_intersect(X1, Y1, X2, Y2):
    W1 = 1000
    H1 = 1000
    W2 = 1000
    H2 = 1000
    if (X1+W1<X2 or X2+W2<X1 or Y1+H1<Y2 or Y2+H2<Y1):
        # print "\t", X1, Y1, "intersect with", X2, Y2
        return False 
    else:
        return True

def check_intersection(arr, X1, Y1):
    if len(arr) == 0:
        return False
    for item in arr:
        X2, Y2 = getXY(item)
        if do_intersect(X1, Y1, X2, Y2):
            return True
    return False

def extract_features(patch_dir, patches):
    # each patch has an associated heatmap
    heatmaps = []

    # this gets the respective heatmaps (we first get a list of all the patches, and then get the heatmaps corresponding to them)
    os.chdir('/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run')
    for patch in patches:
        patch_name = patch.split('/')[-1]

        part1 = patch_name.split('(')[0][:-1]
        num1 = patch_name.split('(')[1].split(',')[0].zfill(10)
        num2 = patch_name.split('(')[1].split(',')[1].split(')')[0].zfill(10)

        patch_name = part1 + '_level0_x' + num1 + '_y' + num2 + '.png.png' #there was a mistake in the naming of files

        # where are the heatmaps stored?
        #prefix = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/mitosis-full_07-07-16/'
        
        full_name = patch_dir + '/' + patch_name

        X1, Y1 = getXY(full_name)

        if os.path.exists(full_name) and check_intersection(heatmaps, X1, Y1) == False and "mask" not in full_name:
            heatmaps.append(full_name)
    
    vector = []
    for threshold_decimal in [0.1, 0.2, 0.3, 0.4, 0.5, 0.67, 0.83, 0.85, 0.91, 0.95, 0.97]: #based on mitko's threhsolds #np.arange(0.1, 1, 0.1): #much slower with larger images
        MANUAL_THRESHOLD = int(255 * threshold_decimal)
        individual_vector = [] 

        mitoses = []
        mitoses_area = []
        mitoses_nonzero = []

        for heatmap in heatmaps:
            try:
                im = imread(heatmap, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                print e
                continue

            thresh = MANUAL_THRESHOLD
            
            _, bw = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)

            # label image regions
            label_image = label(bw)

            A = 0
            potential_mitoses = regionprops(label_image)
            
            for x in potential_mitoses:
                A += x.area

            num_mitoses = len(potential_mitoses) 
            mitoses.append(num_mitoses)

            mitoses_area.append(A)
            
        try:
            # clip to the middle 80 percent
            mitoses = sorted(mitoses)
            if len(mitoses[int(0.1 * len(mitoses)) : int(0.9 * len(mitoses))]) > 1:
                mitoses = mitoses[int(0.1 * len(mitoses)) : int(0.9 * len(mitoses))]
            
            mitoses_area = sorted(mitoses_area)
            if len(mitoses_area[int(0.1 * len(mitoses_area)) : int(0.9 * len(mitoses_area))]) > 1: 
                mitoses_area = mitoses_area[int(0.1 * len(mitoses_area)) : int(0.9 * len(mitoses_area))] 
            
            total_mitoses = sum(i for i in mitoses)
 
            individual_vector.extend([np.median(mitoses), 
                                      np.average(mitoses), 
                                      np.amin(mitoses), 
                                      np.amax(mitoses), 
                                      np.std(mitoses)])

            individual_vector.extend([np.median(mitoses_area), 
                                      np.average(mitoses_area), 
                                      np.amin(mitoses_area), 
                                      np.amax(mitoses_area), 
                                      np.std(mitoses_area)])

        except Exception as e: #0 heatmaps
            print e
            print e.args
            print "There were ", len(heatmaps), " heatmaps."
            individual_vector.append(-1.0)

        # vector is extended at each threshold 
        vector.extend(individual_vector)
    return vector 

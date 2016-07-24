import os
import sys
import subprocess

import numpy as np

from skimage import color
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
        w1_level0 = int(tmp[1][1:-4])

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

def extract_features(patches, gen_heatmaps = 1):
    # each patch has an associated heatmap
  
    heatmaps = []

    # this gets the respective heatmaps (we first get a list of all the patches, and then get the heatmaps corresponding to them)
    os.chdir('/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run')
    for patch in patches:
        patch_name = patch.split('/')[-1]
        # where are the heatmaps stored?
        #prefix = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/mitosis-full_07-07-16/'
        
        # make patch name the new standard
        part1 = patch_name.split('(')[0][:-1]
        num1 = patch_name.split('(')[1].split(',')[0].zfill(10)
        num2 = patch_name.split('(')[1].split(',')[1].split(')')[0].zfill(10)

        patch_name = part1 + '_level0_x' + num1 + '_y' + num2 + '.png'
        prefix = '/data/dywang/Database/Proliferation/evaluation/mitko-fcn-heatmaps-norm/'

        #/data/dywang/Database/Proliferation/evaluation/mitko-fcn-heatmaps-norm/TUPAC-TR-261-(9300,22460).png        
        full_name = prefix + patch_name
        X1, Y1 = getXY(full_name)

        if os.path.exists(full_name) and check_intersection(heatmaps, X1, Y1) == False:
            heatmaps.append(full_name)
    
    vector = []
    for threshold_decimal in np.arange(0, 1, 0.01):
        MANUAL_THRESHOLD = threshold_decimal
        individual_vector = [] 
        tot_mitoses = 0 

        for heatmap in heatmaps: # use 15
            im = color.rgb2gray(imread(heatmap))
            
            thresh = MANUAL_THRESHOLD
            bw = closing(im < thresh, square(3))

            # remove artifacts connected to image border
            cleared = bw.copy()
            clear_border(cleared)

            # label image regions
            label_image = label(cleared)
            borders = np.logical_xor(bw, cleared)
            label_image[borders] = -1

            num_mitoses = len(regionprops(label_image))
            tot_mitoses += num_mitoses

        try:
            individual_vector.append(float(tot_mitoses/(len(heatmaps)))) #scaling for area
        except: #0 heatmaps?
            individual_vector.append(0.0)

        vector.extend(individual_vector) 
    return vector

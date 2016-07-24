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
def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()

def extract_features(patches, gen_heatmaps = 1):
    # each patch has an associated heatmap
  
    heatmaps = []

    # this gets the respective heatmaps (we first get a list of all the patches, and then get the heatmaps corresponding to them)
    os.chdir('/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run')
    for patch in patches:
        patch_name = patch.split('/')[-1]

        part1 = patch_name.split('(')[0][:-1]
        num1 = patch_name.split('(')[1].split(',')[0].zfill(10)
        num2 = patch_name.split('(')[1].split(',')[1].split(')')[0].zfill(10)

        patch_name = part1 + '_level0_x' + num1 + '_y' + num2 + '.png'
        
        # where are the heatmaps stored?
        #prefix = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/mitosis-full_07-07-16/'
        prefix = '/data/dywang/Database/Proliferation/evaluation/mitko-fcn-heatmaps-norm/'   
        
        full_name = prefix + patch_name

        # only look at the patches that have a defined heatmap        
        if os.path.exists(full_name):
            heatmaps.append(full_name)
        
        '''
        elif gen_heatmaps:
            print full_name
            # generate the respective heatmap
            list_file = 'tmp.lst'
            f = open(list_file, 'w')
            f.write('../../../libs/stage03_deepFeatMaps/results/patches_06-29-16/' + patch_name)
            f.close()
            
            subprocess.call("sh /data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run/mitosis-subprocess.sh", shell=True)

            heatmaps.append(full_name)
        '''


    vector = []
    for threshold_decimal in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  0.8, 0.9, 1]:
        ### FOR FULLY CONV HEATMAPS, DON'T SCALE THRESHOLD
        MANUAL_THRESHOLD = threshold_decimal #int(255 * threshold_decimal)
    
        for heatmap in heatmaps[0:15]: # use 15
            individual_vector = []
            
            ### FOR FULLY CONV HEATMAPS, CONVERT TO GRAYSCALE
            im = color.rgb2gray(imread(heatmap))
            
            #thresh = threshold_otsu(im)
            thresh = MANUAL_THRESHOLD
            
            ### FOR FULLY CONV HEATMAPS, IM < THRESH (INVERTED)
            # output heatmaps are inverted, so we need less than thresh
            bw = closing(im < thresh, square(3))

            # remove artifacts connected to image border
            cleared = bw.copy()
            clear_border(cleared)

            # label image regions
            label_image = label(cleared)
            borders = np.logical_xor(bw, cleared)
            label_image[borders] = -1

            num_mitoses = len(regionprops(label_image))
            
            white_pixels = count_nonblack_np(im)

            individual_vector.append(num_mitoses)
            individual_vector.append(white_pixels)

            vector.extend(individual_vector) 
    return vector

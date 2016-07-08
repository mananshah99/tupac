import os
import sys
import subprocess

import numpy as np

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb

def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()

def extract_features(patches, gen_heatmaps = 1):
    # each patch has an associated heatmap
  
    heatmaps = []

    os.chdir('/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run') 
    for patch in patches:
        patch_name = patch.split('/')[-1]
        prefix = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/mitosis-full_06-29-16/'
        full_name = prefix + patch_name
        
        if os.path.exists(full_name):
            heatmaps.append(full_name)
            print full_name
        
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

    from skimage.io import *

    threshold_decimal = 0.75
    MANUAL_THRESHOLD = int(255 * threshold_decimal)
    
    vector = []
    for heatmap in heatmaps:
        individual_vector = []
        im = imread(heatmap)

        white_pixels = count_nonblack_np(im)

        #thresh = threshold_otsu(im)
        thresh = MANUAL_THRESHOLD
        bw = closing(im > thresh, square(3))

        # remove artifacts connected to image border
        cleared = bw.copy()
        clear_border(cleared)

        # label image regions
        label_image = label(cleared)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1

        num_mitoses = len(regionprops(label_image))

        individual_vector.append(num_mitoses)
        individual_vector.append(white_pixels)
   
        vector.extend(individual_vector) 
    return vector

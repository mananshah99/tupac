##
##-- Libraries of interest
##
import sys
import os
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import spams
import copy
import time
import math
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import StainingNormalizer as SN #-- Staining normalizer

def normalize(path):
    img = cv2.imread(path) #-- Read image
    img[:,:,(0,2)] = img[:,:,(2,0)] #-- RGB representation
    T = SN.Controller(img)

    Macenko_params = SN.method.macenko_p 
    Macenko_decomposition = T(SN.method.macenko)
    
    norm_img, a, b, c = Macenko_decomposition

    abspath = path.replace('patches_07-14-16', 'patches_07-14-16-norm')
    cv2.imwrite(abspath, norm_img)

import os

# Preprocess the total files sizes
sizecounter = 0
for dirpath, dirs, files in tqdm(os.walk('patches_07-14-16')):
    for filename in files:
        if "mask" not in filename:
            sizecounter += 1

print "TOTAL: ", sizecounter
# Load tqdm with size counter instead of files counter
with tqdm(total=sizecounter) as pbar:
    for root, dirs, files in os.walk('patches_07-14-16'):
        for f in files:
            p = os.path.join(root, f)
            if "mask" not in p:
                abspath = os.path.abspath(p) 
                normalize(abspath)
                pbar.update(1)


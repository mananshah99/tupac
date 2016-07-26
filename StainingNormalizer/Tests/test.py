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

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import StainingNormalizer as SN #-- Staining normalizer

def normalize(path):
    img = cv2.imread(path) #-- Read image
    img[:,:,(0,2)] = img[:,:,(2,0)] #-- RGB representation
    T = SN.Controller(img)

    Macenko_params = SN.method.macenko_p 
    Macenko_decomposition = T(SN.method.macenko)
    
    norm_img, a, b, c = Macenko_decomposition

    cv2.imwrite('testout.png', norm_img[:,:,(2,1,0)])

normalize('example1.tif') 

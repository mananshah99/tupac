import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def count_mitoses(heatmap,
                  blur_radius = 1.0,
                  threshold = 50,
                  plot = False):

    img = scipy.misc.imread(heatmap)
 
    # smooth the image (to remove small objects)
    imgf = ndimage.gaussian_filter(img, blur_radius)

    # find connected components
    labeled, nr_objects = ndimage.label(imgf > threshold) 

    if plot:
        plt.imsave('temp.png', labeled, cmap='gray')
        print "Saved plot"

    return nr_objects, labeled

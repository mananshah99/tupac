import os
import sys
sys.path.insert(0, '../METHOD3/external/caffe/python')
import subprocess

import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.io import *
import caffe

def in_bounds(coordinate, mx = 1000):
    if coordinate < 0:
        coordinate = 0
    if coordinate > mx:
        coordinate = mx
    return coordinate

def extract_features(net, transformer, patches, gen_heatmaps = 1):
    # each patch has an associated heatmap
  
    heatmaps = []
    real_patches = []

    # this gets the respective heatmaps (we first get a list of all the patches, and then get the heatmaps corresponding to them)
    os.chdir('/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run')
    for patch in patches:
        patch_name = patch.split('/')[-1]
        prefix = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/mitosis-full_07-07-16/'
        full_name = prefix + patch_name

        # only look at the patches that have a defined heatmap        
        if os.path.exists(full_name):
            heatmaps.append(full_name)
            real_patches.append(patch)
 
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
    for i, heatmap in enumerate(heatmaps[0:15]): # use 15
        individual_vector = []
        im = imread(heatmap)
        associated_patch = imread(real_patches[i])

        thresh = 0.9
        bw = closing(im > thresh, square(3))

        # remove artifacts connected to image border
        cleared = bw.copy()
        clear_border(cleared)

        # label image regions
        label_image = label(cleared)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1

        for region in regionprops(label_image):
            y0,x0 = region.centroid
            region_im = associated_patch[in_bounds(y0-112):in_bounds(y0+112), in_bounds(x0-112):in_bounds(x0+112)]
            
            try:
                transformed_image = transformer.preprocess('data', caffe.io.resize_image(region_im, (224, 224)))
            except Exception as e:
                print associated_patch.shape
                print label_image.shape 
                print e
                continue
            batch = np.array(transformed_image)
            net.blobs['data'].data[...] = batch

            output = net.forward()
            features = net.blobs['pool5/7x7_s1'].data.copy()
            
            # convert the features to a readable format
            tmp = []
            for x in features[0]:
                y = x[0][0]
                tmp.append(y)
                
            individual_vector.extend(tmp)
        vector.extend(individual_vector) 
    return vector

import os
import sys
import cv2
sys.path.insert(0, '../METHOD3/external/caffe/python')
import subprocess
import openslide as osi
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
from tqdm import tqdm

levelpow = 4

def extend_inds_to_level0(input_level, h, w):
    gap = input_level - 0
    v = np.power(levelpow, gap)
    hlist = h * v + np.arange(v)
    wlist = w * v + np.arange(v)
    hw = []
    for hv in hlist:
        for wv in wlist:
            hw.append([hv, wv])
    return hw

def get_tl_pts_in_level0(OUT_LEVEL, h_level0, w_level0, wsize):
    scale = np.power(levelpow, OUT_LEVEL)
    wsize_level0 = wsize * scale
    wsize_level0_half = wsize_level0 / 2

    h1_level0, w1_level0 = h_level0 - wsize_level0_half, w_level0 - wsize_level0_half
    return int(h1_level0), int(w1_level0)

def get_image(wsi, h1_level0, w1_level0, OUT_LEVEL, wsize):
    img = wsi.read_region(
            (w1_level0, h1_level0),
            OUT_LEVEL, (wsize, wsize))
    img = np.asarray(img)[:,:,:3]
    return img


def extract_features(net, transformer, heatmap):
    ALL_BLACK = False
    slide = None
    try:
        im_name = heatmap.split('/')[-1].split('.')[0]
        print heatmap
        slide = osi.OpenSlide('/data/dywang/Database/Proliferation/data/TrainingData/training_image_data/' + im_name + '.svs')
        im = cv2.imread(heatmap, cv2.IMREAD_GRAYSCALE)
        print im.shape
    except Exception as e:
        print e
        print e.args
        ALL_BLACK = True

    vector = []
    for threshold_decimal in [0.7]: 

        thresh = int(255 * threshold_decimal)
        individual_vector = []

        if ALL_BLACK:
            individual_vector.extend([0]*200)
            vector.extend(individual_vector)
            continue

        _, bw = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)

        # label image regions
        label_image = label(bw)

        potential_mitoses = regionprops(label_image)

        for region in potential_mitoses:
            centroid = region.centroid
            indices = extend_inds_to_level0(2, centroid[0], centroid[1])

            idx = int(len(indices) / 2)
            chcw_level0 = indices[idx] # take middle point (this is basically the centroid at level 0)
            h_level0, w_level0 = chcw_level0
            h1_level0, w1_level0 = get_tl_pts_in_level0(0,
                                                        h_level0,
                                                        w_level0,
                                                        224) # should give top left corner of patch

            associated_patch = get_image(slide, h1_level0, w1_level0, 0, 224)
 
            associated_patch = np.asarray(associated_patch)[:,:,:3]
            try:
                transformed_image = transformer.preprocess('data', caffe.io.resize_image(associated_patch, (224, 224)))
            except Exception as e:
                print e
                print e.args
                continue
            
            batch = np.array(transformed_image)
            net.blobs['data'].data[...] = batch

            output = net.forward()
            features = net.blobs['ip1'].data.copy()
            # convert the features to a readable format
            tmp = []
            print features.shape
            for x in features[0]:
                y = []
                for q in xrange(features.shape[2]):
                    y.append(x[0][q])
                tmp.extend(y)

            individual_vector.extend(tmp)
        vector.extend(individual_vector)
    return vector

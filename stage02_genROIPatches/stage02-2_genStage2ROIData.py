# Stage 2 ROI
import numpy as np
import sys
import random
import os
import cv2
import openslide as osi

levelpow = 4
input_level = 2
output_level = 0

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

def keep_in_bounds(coordinate, patch_size=256):
    if coordinate < int(patch_size/2):
        coordinate = int(patch_size/2)

    return coordinate

def create_patch(img, center_pixel, patch_size):
    center_x = center_pixel[0]
    center_y = center_pixel[1]

    halfway = int(float(patch_size)/2)

    rect = img[center_y - halfway : center_y + halfway, center_x - halfway : center_x + halfway]
    return rect

def nearest_neighbors(im, i, j, d=4):
    n = im[i-d:i+d+1, j-d:j+d+1]
    return n

# ROI level 1 is done on level 1 small images
thresholded_roi = [f for f in os.listdir('/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16/thresholded-0.85/')]

patch_size = 256
n_patches_per = 256

# the initial list (with 0 = non ROI and 1 = ROI) is located here:
#   ../../train/ROI_stage01_LEVEL00/imglist_stage01.lst
#

for f in thresholded_roi:
    print f
    roi_locations = cv2.imread('/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16/thresholded-0.85/' + f, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    original_image = cv2.imread('/data/dywang/Database/Proliferation/data/TrainingData/small_images-level-2-mask/' + f)
    wsi_image = osi.OpenSlide('/data/dywang/Database/Proliferation/data/TrainingData/training_image_data/' + f[:-4] + '.svs')
    
    w = original_image.shape[0]
    h = original_image.shape[1]
    for i in xrange(n_patches_per):
        internalct = 0
        while True:
            x_rand = random.randint(129,h-129)
            y_rand = random.randint(129,w-129)
            # pixel value is im[y_rand, x_rand]
        
            nn = nearest_neighbors(roi_locations, y_rand, x_rand, 128)
            if any(255 in sublist for sublist in nn):
                internalct += 1
            else:
                # each centroid is a tuple
                indices = extend_inds_to_level0(input_level, y_rand, x_rand)
                idx = int(len(indices) / 2)
                chcw_level0 = indices[idx] # take middle point (this is basically the centroid at level 0)
                h_level0, w_level0 = chcw_level0
                h1_level0, w1_level0 = get_tl_pts_in_level0(output_level,
                                                        h_level0,
                                                        w_level0,
                                                        patch_size) # should give top left corner of patch

                img = get_image(wsi_image, h1_level0, w1_level0, output_level, 256)

                cv2.imwrite('ROI-Stage2' + '/' + f + '-(' + str(h1_level0) + ',' + str(w1_level0) + ').png', img)
                break

            if internalct >= 50:
                break

# The purpose of this script is to generate positive (mitotic) patches
# and negative (nuclei but non mitotic) patches within each training image.
# These will be fed into a caffe model and trained via GoogleNet.

import numpy as np
import cv2
import csv, os
from skimage.measure import label, regionprops
import random
from scipy.spatial import distance
#def keep_in_bounds(coordinate, patch_size=100):
#    if coordinate < int(patch_size/2):
#        coordinate = int(patch_size/2)
#    return coordinate

def getFolders(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    return fpath

def nearest_neighbors(im, i, j, d=4):
    n = im[i-d:i+d+1, j-d:j+d+1]
    return n

def create_patch(img, center_pixel, patch_size):
    center_x = center_pixel[0]
    center_y = center_pixel[1]
    h, w, c = img.shape

    halfway = int(float(patch_size)/2)
    if center_y - halfway >= 0 and center_y + halfway < h and center_x - halfway >= 0 and center_x + halfway < w:
        rect = img[center_y - halfway : center_y + halfway, center_x - halfway : center_x + halfway]
    else:
        rect = None
    return rect

def gen_patches(img_path, csv_path, patch_size = 100, output_directory = "", nn_patch_size = 10, n_patches=10, patch_frac_denominator=4):
    ground_truth_points = load_pts(csv_path)
    print "#mitosis = %d"%(len(ground_truth_points))
    image = cv2.imread(img_path)
    msk_image = cv2.imread(msk_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    itms = img_path.split('/')
    img_name = itms[-1]
    img_name_root = img_name.split('.')[0]
    wsi_name = itms[-2]
    output_directory_for_img = getFolders('%s/%s/%s'%(output_directory, wsi_name, img_name_root))

    for center_pixel in ground_truth_points:
        initial_x = center_pixel[0]
        initial_y = center_pixel[1]

        center_patch = create_patch(image, (initial_x, initial_y), patch_size) # return None if the patch is out of image.
        output_image_path_name = '%s/%s_%s_x%010d_y%010d.png'%(output_directory_for_img, wsi_name, img_name_root, initial_x, initial_y)
        if center_patch is not None:
            if msk_image[initial_y, initial_x] > 0:
                print "\t\t -> Saved center crop to %s (y:%d x:%d)"%(output_image_path_name, initial_y, initial_x)
                cv2.imwrite(output_image_path_name, center_patch)

                neighbors = nearest_neighbors(msk_image, initial_y, initial_x, nn_patch_size)
                #output_image_path_name = '%s/%s_%s_x%010d_y%010d_mask.png'%(output_directory_for_img, wsi_name, img_name_root, initial_x, initial_y)
                #cv2.imwrite(output_image_path_name, neighbors)

                for i in range(1, n_patches):
                    # what we really need is to get other points within the contour
                    count = 0
                    while True:
                        x = random.choice(np.arange(neighbors.shape[0]))
                        y = random.choice(np.arange(neighbors.shape[0]))
                        if neighbors[y,x] == 255:
                            new_x = initial_x - (nn_patch_size - x) # !!!!
                            new_y = initial_y - (nn_patch_size - y)
                            shifted_patch = create_patch(image, (new_x, new_y), patch_size)
                            if shifted_patch is not None:
                                output_image_path_name = '%s/%s_%s_x%010d_y%010d.png'%(output_directory_for_img, wsi_name, img_name_root, new_x, new_y)
                                cv2.imwrite(output_image_path_name, shifted_patch)
                                print "\t\t -> Saved shifted crop to %s (x:%d x:%d)"%(output_image_path_name, initial_y, initial_x)
                            break
                        if count >= 500:
                            print "[!] Couldn't find a valid patch"
                            break
                        count += 1
            else: # the ground_truth's mask is ZERO ! Check it !!!!
                print "\t\t -> ERROR: missing the ground truth: " + output_image_path_name
        else:
            print "\t\t -> WARN: cropped region is out of image: " + output_image_path_name

def load_pts(csv_file):
    pts = []
    if not os.path.exists(csv_file):
        print "Missing", csv_file
        return pts
    else:
        with open(csv_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            for line in spamreader:
                row, col = line[0], line[1]
                pts.append((int(col), int(row))) 
        return pts

def notMitosis(pt, gpts, r = 30):
    dis = []
    for gpt in gpts:
        dis.append(distance.euclidean(pt, gpt))
    minv = np.min(dis)
    if minv < r:
        return False
    else:
        return True

def nearest_neighbors(im, i, j, d=4):
    n = im[i-d:i+d+1, j-d:j+d+1]
    return n

IMG_DATA_ROOT = '../../../../data/mitoses/mitoses_image_data_cn'
MSK_DATA_ROOT = '../../../../data/mitoses/mitoses_image_data'
CSV_DATA_ROOT = '../../../../data/mitoses/mitoses_ground_truth'

HEP_DATA_ROOT = '../step03_getHeatmap/mnet/mnet_stage01/heatmap'

def get_false_negative(img_path, msk_path, hep_path, csv_path, mitosis_radius=20, num_neg=200, patch_size=100, output_directory = ''):
    ground_truth_points = load_pts(csv_path)
    if 0:
        image = cv2.imread(img_path)
        msk_image = cv2.imread(msk_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        print hep_path
        hep_image = cv2.imread(hep_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        # remove mitosis
        for center_pt in ground_truth_points:
            hep_image[center_pt[1] - mitosis_radius : center_pt[1] + mitosis_radius, center_pt[0] - mitosis_radius : center_pt[0] + mitosis_radius] = 0

        itms = img_path.split('/')
        img_name = itms[-1]
        img_name_root = img_name.split('.')[0]
        wsi_name = itms[-2]
        output_directory_for_img = getFolders('%s/%s/%s'%(output_directory, wsi_name, img_name_root))
        
        threshold = 0.8 * 255
        ys, xs = np.where(hep_image > threshold)
        neg_pts = [(y,x) for y, x in zip(ys, xs)]
        print "#pts=%d"%(len(neg_pts))
        random.seed(1122)
        num_neg_size = np.min((num_neg, len(neg_pts)))
        neg_pts_sel = random.sample(neg_pts, num_neg_size)
        for neg_pt in neg_pts_sel:
            y, x = neg_pt
            center_patch = create_patch(image, (y, x), patch_size) # return None if the patch is out of image.
            output_image_path_name = '%s/%s_%s_x%010d_y%010d.png'%(output_directory_for_img, wsi_name, img_name_root, x, y)
            print "\t\t -> Saved center crop to %s (y:%d x:%d)"%(output_image_path_name, y, x)
            if center_patch is not None:
                cv2.imwrite(output_image_path_name, center_patch)

for line in [l.strip() for l in open('all_image.lst').readlines()]:
    wsi_name, csv_name = line.split('/')
    name_root = csv_name.split('.')[0]

    img_path = '%s/%s/%s.tif_nc.png'%(IMG_DATA_ROOT, wsi_name, name_root)
    csv_path = '%s/%s/%s.csv'%(CSV_DATA_ROOT, wsi_name, name_root)
    msk_path = '%s/%s/%s_mask_nuclei.png'%(MSK_DATA_ROOT, wsi_name, name_root)
    hep_path = '%s/%s/%s.tif.png'%(HEP_DATA_ROOT, wsi_name, name_root)
#    if 0: # get positive
#       gen_patches(img_path, csv_path, n_patches=40, nn_patch_size = 10, output_directory='training_examples_s2/pos')
    if 1: # get false positive
       get_false_negative(img_path, msk_path, hep_path, csv_path, output_directory='training_examples_s2/neg')

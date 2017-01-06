# The purpose of this script is to generate positive (mitotic) patches
# and negative (nuclei but non mitotic) patches within each training image.
# These will be fed into a caffe model and trained via GoogleNet.

import numpy as np
import cv2
import csv, os
from skimage.measure import label, regionprops
import random
from scipy.spatial import distance


def load_pts(csv_file):
    pts = []
    with open(csv_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for line in spamreader:
            row, col = line[0], line[1]
            pts.append((int(col), int(row)))
    return pts

IMG_DATA_ROOT = '/home/dywang/Proliferation/data/mitoses/mitoses_image_data'
CSV_DATA_ROOT = '/home/dywang/Proliferation/data/mitoses/mitoses_ground_truth'
HEP_DATA_ROOT = '/home/dywang/Proliferation/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/heatmap_mc13_tr/heatmap/mitoses_image_data_cn'

radius = 10
for line in [l.strip() for l in open('/home/dywang/Proliferation/data/mitoses/all_image_mc13_tr.lst').readlines()]:
    wsi_name, csv_name = line.split('/')
    name_root = csv_name.split('.')[0]

    img_path = '%s/%s/%s.tif'%(IMG_DATA_ROOT, wsi_name, name_root)
    csv_path = '%s/%s/%s.csv'%(CSV_DATA_ROOT, wsi_name, name_root)
    msk_path = '%s/%s/%s_mask_nuclei.png'%(IMG_DATA_ROOT, wsi_name, name_root)
    hep_path = '%s/%s/%s.tif_nc.png.png'%(HEP_DATA_ROOT, wsi_name, name_root)
    label_path = '%s/%s/%s_label.png'%(HEP_DATA_ROOT, wsi_name, name_root)
    
    if os.path.exists(csv_path):
        ground_truth_points = load_pts(csv_path)
    else:
        ground_truth_points = []

    print img_path, msk_path
    msk_image = cv2.imread(msk_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    hep_image = cv2.imread(hep_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    print np.min(msk_image), np.max(msk_image)

    lab_image = np.zeros(msk_image.shape, np.uint8)
    
    # add negative samples
    lab_image[msk_image > 0] = 1 
    
    # add false positive samples
    lab_image[hep_image > 0.5 * 255] = 3 

    # add pos samples
    for point in ground_truth_points:
        lab_image[point[1]-radius: point[1]+radius, point[0]-radius:point[0]+radius] = 2

    cv2.imwrite(label_path, lab_image)
#    break

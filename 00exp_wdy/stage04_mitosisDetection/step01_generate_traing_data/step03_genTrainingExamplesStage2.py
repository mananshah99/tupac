# The purpose of this script is to generate positive (mitotic) patches
# and negative (nuclei but non mitotic) patches within each training image.
# These will be fed into a caffe model and trained via GoogleNet.

import numpy as np
import cv2
import csv, os
from skimage.measure import label, regionprops
import random
from scipy.spatial import distance
import skimage.io as skio
from skimage.color import label2rgb
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

IMG_DATA_ROOT = 'mitoses/mitoses_image_data_cn3'
MSK_DATA_ROOT = 'mitoses/mitoses_image_data'
CSV_DATA_ROOT = 'mitoses/mitoses_ground_truth'

#HEP_DATA_ROOT = '../step03_getHeatmap/mnet/mnet_stage01/heatmap'
HEP_DATA_ROOT = '../step03_getHeatmap/mnet_v2/stage01'

def get_false_negative_v1(img_path, msk_path, hep_path, csv_path, mitosis_radius=30, num_neg=200, patch_size=100, output_directory = ''):
    print "Input Image: ", img_path
    ground_truth_points = load_pts(csv_path)
    if 1:
        image = cv2.imread(img_path)
        msk_image = cv2.imread(msk_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        print hep_path
        hep_image = cv2.imread(hep_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        for center_pt in ground_truth_points:
            hep_image[center_pt[1] - mitosis_radius : center_pt[1] + mitosis_radius, center_pt[0] - mitosis_radius : center_pt[0] + mitosis_radius] = 0
            msk_image[center_pt[1] - mitosis_radius : center_pt[1] + mitosis_radius, center_pt[0] - mitosis_radius : center_pt[0] + mitosis_radius] = 0

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
        print "neg_pts_sel=", len(neg_pts_sel)
        #if num_neg_size < num_neg:
        #    ys, xs = np.where(msk_image > 0)
        #    neg_pts = [(y,x) for y, x in zip(ys, xs)]
        #    neg_pts_sel += random.sample(neg_pts, num_neg - num_neg_size)

        for neg_pt in neg_pts_sel:
            y, x = neg_pt
            center_patch = create_patch(image, (y, x), patch_size) # return None if the patch is out of image.
            output_image_path_name = '%s/%s_%s_x%010d_y%010d.png'%(output_directory_for_img, wsi_name, img_name_root, x, y)
            if center_patch is not None:
                cv2.imwrite(output_image_path_name, center_patch)

def get_false_negative_v2(img_path, msk_path, hep_path, csv_path, mitosis_radius=30, num_neg=200, patch_size=100, output_directory = ''):
    print "Input Image: ", img_path
    ground_truth_points = load_pts(csv_path)
    if len(ground_truth_points) == 0:
        print "ERROR: no ground truth"
    if 1:
        image = cv2.imread(img_path)
        msk_image = cv2.imread(msk_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        print hep_path
        hep_image = cv2.imread(hep_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        for center_pt in ground_truth_points:
            hep_image[center_pt[1] - mitosis_radius : center_pt[1] + mitosis_radius, center_pt[0] - mitosis_radius : center_pt[0] + mitosis_radius] = 0
            msk_image[center_pt[1] - mitosis_radius : center_pt[1] + mitosis_radius, center_pt[0] - mitosis_radius : center_pt[0] + mitosis_radius] = 0

        itms = img_path.split('/')
        img_name = itms[-1]
        img_name_root = img_name.split('.')[0]
        wsi_name = itms[-2]
        output_directory_for_img = getFolders('%s/%s/%s'%(output_directory, wsi_name, img_name_root))

        threshold = 0.2 * 255
        bw = hep_image > threshold
        label_bw = label(bw)
        bw_label_bw = label2rgb(label_bw, image=bw)

        skio.imsave('%s/bw.jpg'%(output_directory_for_img), bw.astype(np.uint8) * 255)
        skio.imsave('%s/bw_label.jpg'%(output_directory_for_img), bw_label_bw)
        props = regionprops(label_bw)

        step_size = 5
        print "#prop =", len(props)
        for prop_id, prop in enumerate(props):
            (min_row, min_col, max_row, max_col) = prop['bbox']
            bbox_h = max_row - min_row + 1
            bbox_w = max_col - min_col + 1
            for w1 in np.arange(min_col + (step_size-1)/2, max_col, step_size):
                for h1 in np.arange(min_row + (step_size-1)/2, max_row, step_size):
                    center_patch = create_patch(image, (h1, w1), patch_size) # return None if the patch is out of image.
                    output_image_path_name = '%s/%s_%s_x%010d_y%010d.png'%(output_directory_for_img, wsi_name, img_name_root, w1, h1)
                    #print "\t\t", output_image_path_name
                    if center_patch is not None:
                        cv2.imwrite(output_image_path_name, center_patch)

def get_false_negative_v3(img_path, msk_path, hep_path, csv_path, mitosis_radius=30, num_neg=200, patch_size=100, output_directory = ''):
    print "Input Image: ", img_path
    ground_truth_points = load_pts(csv_path)
    if len(ground_truth_points) == 0:
        print "ERROR: no ground truth"
    if 1:
        image = cv2.imread(img_path)
        msk_image = cv2.imread(msk_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        print hep_path
        hep_image = cv2.imread(hep_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        for center_pt in ground_truth_points:
            hep_image[center_pt[1] - mitosis_radius : center_pt[1] + mitosis_radius, center_pt[0] - mitosis_radius : center_pt[0] + mitosis_radius] = 0
            msk_image[center_pt[1] - mitosis_radius : center_pt[1] + mitosis_radius, center_pt[0] - mitosis_radius : center_pt[0] + mitosis_radius] = 0

        itms = img_path.split('/')
        img_name = itms[-1]
        img_name_root = img_name.split('.')[0]
        wsi_name = itms[-2]
        output_directory_for_img = getFolders('%s/%s/%s'%(output_directory, wsi_name, img_name_root))

        threshold = 0.2 * 255
        bw = hep_image > threshold
        label_bw = label(bw)
        bw_label_bw = label2rgb(label_bw, image=bw)

        skio.imsave('%s/bw.jpg'%(output_directory_for_img), bw.astype(np.uint8) * 255)
        skio.imsave('%s/bw_label.jpg'%(output_directory_for_img), bw_label_bw)
        props = regionprops(label_bw)

        step_size = 5
        print "#prop =", len(props)
        for prop_id, prop in enumerate(props):
            (min_row, min_col, max_row, max_col) = prop['bbox']
            bbox_h = max_row - min_row + 1
            bbox_w = max_col - min_col + 1
            for w1 in np.arange(min_col + (step_size-1)/2, max_col, step_size):
                for h1 in np.arange(min_row + (step_size-1)/2, max_row, step_size):
                    center_patch = create_patch(image, (h1, w1), patch_size) # return None if the patch is out of image.
                    output_image_path_name = '%s/%s_%s_x%010d_y%010d.png'%(output_directory_for_img, wsi_name, img_name_root, w1, h1)
                    #print "\t\t", output_image_path_name
                    if center_patch is not None:
                        cv2.imwrite(output_image_path_name, center_patch)

def get_neg_samples(line):
    folder_name, img_name = line.split('/')
    img_name_root = img_name.split('.')[0]

    img_path = '%s/%s/%s_Normalized.tif'%(IMG_DATA_ROOT, folder_name, img_name_root)
    csv_path = '%s/%s/%s.csv'%(CSV_DATA_ROOT, folder_name, img_name_root)
    msk_path = '%s/%s/%s_mask_nuclei.png'%(MSK_DATA_ROOT, folder_name, img_name_root)
    hep_path = '%s/%s/%s_Normalized.tif.png'%(HEP_DATA_ROOT, folder_name, img_name_root)
    if 1: # get false positive
       get_false_negative_v2(img_path, msk_path, hep_path, csv_path, patch_size=64, num_neg=100, output_directory='training_examples_s2/neg')
    #break

from multiprocessing import Pool

if __name__ == "__main__":
    lines = [l.strip() for l in open('all_image.lst').readlines()]
    if 1:
        for line in lines:
            get_neg_samples(line)
            break
    else:
        pool = Pool(20)
        pool.map(get_neg_samples, lines)
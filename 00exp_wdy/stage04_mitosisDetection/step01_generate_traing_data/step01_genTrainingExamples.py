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

# Will generate square patches only (patch_size x patch_size)
# center_pixel is guaranteed to be in each patch
# center_pixel is (x, y)
def gen_patches(img_path, # something.tif
                center_pixel, # a tuple
                mask_image, # mask for contours
                patch_size = 100,
                output_directory = "",
                nn_patch_size = 10,
                n_patches=10,
                patch_frac_denominator=4):

    itms = img_path.split('/')
    img_name = itms[-1]
    img_name_root = img_name.split('.')[0]
    wsi_name = itms[-2]

    output_directory_for_img = getFolders('%s/%s/%s'%(output_directory, wsi_name, img_name_root))

    image = cv2.imread(img_path)

    initial_x = center_pixel[0]
    initial_y = center_pixel[1]

    center_patch = create_patch(image, (initial_x, initial_y), patch_size) # return None if the patch is out of image.
    output_image_path_name = '%s/%s_%s_x%010d_y%010d.jpg'%(output_directory_for_img, wsi_name, img_name_root, initial_x, initial_y)
    if center_patch is not None:
        if msk_image[initial_y, initial_x] > 0:
            print "\t\t -> Saved center crop to " + output_image_path_name
            cv2.imwrite(output_image_path_name, center_patch)

            neighbors = nearest_neighbors(mask_image, initial_y, initial_x, nn_patch_size)

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
                            output_image_path_name = '%s/%s_%s_x%010d_y%010d.jpg'%(output_directory_for_img, wsi_name, img_name_root, new_x, new_y)
                            cv2.imwrite(output_image_path_name, shifted_patch)
                            print "\t\t -> Saved shifted crop to " + output_image_path_name
                        break
                    if count >= 500:
                        print "[!] Couldn't find a valid patch"
                        break
                    count += 1
            return True
        else: # the ground_truth's mask is ZERO ! Check it !!!!
            print "\t\t -> ERROR: missing the ground truth: " + output_image_path_name
            return False
    else:
        print "\t\t -> WARN: cropped region is out of image: " + output_image_path_name
        return False

def load_pts(csv_file):
    pts = []
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

IMG_DATA_ROOT = '../../../../data/mitoses/mitoses_image_data_cn'
MSK_DATA_ROOT = '../../../../data/mitoses/mitoses_image_data'
CSV_DATA_ROOT = '../../../../data/mitoses/mitoses_ground_truth'
for line in [l.strip() for l in open('groundtruth.lst').readlines()]:
    wsi_name, csv_name = line.split('/')
    name_root = csv_name.split('.')[0]

    img_path = '%s/%s/%s.tif_nc.png'%(IMG_DATA_ROOT, wsi_name, name_root)
    csv_path = '%s/%s/%s.csv'%(CSV_DATA_ROOT, wsi_name, name_root)
    msk_path = '%s/%s/%s_mask_nuclei.png'%(MSK_DATA_ROOT, wsi_name, name_root)
    ground_truth_points = load_pts(csv_path)
    print img_path

    msk_image = cv2.imread(msk_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    if 1:
        for point in ground_truth_points:
            print "\tPositive => " + "(" + str(point[0]) + "," + str(point[1]) + ")"
            gen_patches(img_path, point, msk_image, patch_size=64, n_patches = 10, nn_patch_size = 5, output_directory='training_examples/pos')
    if 0:
        cnts, _ = cv2.findContours(msk_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ct, NegNUM = 0, 50
        random.shuffle(cnts)
        for cnt in cnts:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            tup = (cX, cY)
            if notMitosis(tup, ground_truth_points):
                print "\tNegative => " + "(" + str(tup[0]) + "," + str(tup[1]) + ")"
                r = gen_patches(img_path, tup, msk_image, patch_size=64, output_directory='training_examples/neg', n_patches=1, nn_patch_size = 5)
                if r:
                    ct = ct + 1
                    if msk_image.shape[0] > 2000: # for large images
                        if ct == NegNUM * 4:
                            break
                    else:
                        if ct == NegNUM:
                            break
            else:
                print "\t\t -> Error: too closed to MITOSIS: %d %d"%(cX, cY)

# The purpose of this script is to generate positive (mitotic) patches
# and negative (nuclei but non mitotic) patches within each training image.
# These will be fed into a caffe model and trained via GoogleNet.

import numpy as np
import cv2
import csv
from skimage.measure import label, regionprops
import random

def keep_in_bounds(coordinate, patch_size=100):
    if coordinate < int(patch_size/2):
        coordinate = int(patch_size/2) 

    return coordinate

def nearest_neighbors(im, i, j, d=4):
    n = im[i-d:i+d+1, j-d:j+d+1]
    return n


def find_false_negatives(image_name, ground_truth_points):
    name = image_path.split('mitoses_image_data/')[1]
    name = name.replace('/', '-')[:-4]

    mask = image_path[:-4] + '_mask_nuclei.png'
    mask_image = cv2.imread(mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    mitosis_mask = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/mitosis-train_07-07-16/' + name + '.png'
    mitosis_mask_image = cv2.imread(mitosis_mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    image = cv2.imread(image_name)
    thresh = 0.5 * 255

    positive_y, positive_x = (mitosis_mask_image > thresh).nonzero()

    ct = 0
    for p in ground_truth_points:
        center_x = p[0]
        center_y = p[1]        
        radius = 20  # patches are 100 x 100

        # generate a rectangle, we'll get the circle from this rectangle
        potential_x_values = range(center_x - radius, center_x + radius)
        potential_y_values = range(center_y - radius, center_y + radius)
        
        for i in range(0, len(potential_x_values)):
            x = potential_x_values[i]
            y = potential_y_values[i]
            
            x = keep_in_bounds(x)
            y = keep_in_bounds(y)

            if (x-center_x)**2 + (y-center_y)**2 <= radius**2:
                try:
                    if mask_image[y,x] > 0 or (x-center_x)**2 + (y-center_y)**2 <= 5**2: # part of the nucleus, is a mitosis
                        if mitosis_mask_image[y,x] == 0: # didn't catch it in the mitosis mask
                            # false negative
                            patch = create_patch(image, (x, y), 100) # patch_size = 100
                            cv2.imwrite('training_examples/pos_stage2' + '/' + name + '-(' + str(x) + ',' + str(y) + ').png', patch)
                            ct += 1
                except Exception as e:
                    print e
 
    print "False negs: ", ct

# generate patches from > 0.5 on heatmap but not mitotic
# these patches are saved in the neg/ directory
def gen_augmented_patches(image_name, ground_truth_points):
    name = image_path.split('mitoses_image_data/')[1]
    name = name.replace('/', '-')[:-4]

    mask = image_path[:-4] + '_mask_nuclei.png'
    mask_image = cv2.imread(mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    mitosis_mask = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/mitosis-train_07-07-16/' + name + '.png'
    mitosis_mask_image = cv2.imread(mitosis_mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)    

    image = cv2.imread(image_name)
    thresh = 0.5 * 255
   
    positive_y, positive_x = (mitosis_mask_image > thresh).nonzero()

    false_pos = 0
    true_pos = 0

    skipped = False
    last_y_used = positive_y[0]
    last_x_used = positive_x[0]

    for i in range(0, len(positive_x)):
        x = positive_x[i]
        y = positive_y[i]

        # row-major order
        if i >= 1:
            if x <= last_x_used + 8 and last_y_used == y: # step size = 4, make sure to check we are on the same row
                continue
            else:
                last_x_used = x
                last_y_used = y

        x = keep_in_bounds(x)
        y = keep_in_bounds(y)

        truePositive = False
        
        for index, p in enumerate(ground_truth_points):
            center_x = p[0]
            center_y = p[1]
            radius = 40  # patches are 100 x 100

            if (x-center_x)**2 + (y-center_y)**2 <= radius**2:        
                # inside a mitotic circle
                # true positive
                truePositive = True
                break
        
        if truePositive == True:
            true_pos += 1
        else:
            patch = create_patch(image, (x, y), 100) # patch_size = 100
            cv2.imwrite('training_examples/neg_stage2' + '/' + name + '-(' + str(x) + ',' + str(y) + ').png', patch)
            false_pos += 1

    print false_pos, true_pos

def create_patch(img, center_pixel, patch_size):
    center_x = center_pixel[0]
    center_y = center_pixel[1]

    halfway = int(float(patch_size)/2)

    rect = img[center_y - halfway : center_y + halfway, center_x - halfway : center_x + halfway]
    return rect

def load_pts(csv_file):
    pts = []
    with open(csv_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            pts.append((int(row[1]), int(row[0])))

    return pts

# with all images in images.lst
ground_truth = open('groundtruth.lst')
image_paths = open('images.lst')

ground_truth_lines = ground_truth.readlines()
ground_truth.close()

#  mitoses_ground_truth/02/23.csv
#  ../../data/mitoses/mitoses_image_data/02/20.tif
for i in range(0, 587): # really just all of the lines, cut down for testing
    line = ground_truth_lines[i].strip('\n')
    image_path = '../../data/mitoses/mitoses_image_data' + line[-10:-4] + '.tif'
    ground_truth_points = load_pts(line)
    print image_path
    find_false_negatives(image_path, ground_truth_points)
#    gen_augmented_patches(image_path, ground_truth_points)

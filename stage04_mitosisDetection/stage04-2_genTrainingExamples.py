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

# Will generate square patches only (patch_size x patch_size)
# center_pixel is guaranteed to be in each patch
# center_pixel is (x, y)
def gen_patches(image_name, # something.tif
                center_pixel, # a tuple
                mask_image, # mask for contours
                patch_size = 100,
                output_directory = "",
                nn_patch_size = 15,
                n_patches=10,
                patch_frac_denominator=4):

    name = image_path.split('mitoses_image_data/')[1]
    name = name.replace('/', '-')[:-4]

    image = cv2.imread(image_name)

    initial_x = center_pixel[0]
    initial_y = center_pixel[1]

    initial_x = keep_in_bounds(initial_x)
    initial_y = keep_in_bounds(initial_y)

    center_patch = create_patch(image, (initial_x, initial_y), patch_size)

    cv2.imwrite(output_directory + '/' + name  + '-(' + str(initial_x) + ',' + str(initial_y) + ').png', center_patch)

    print "\t\t -> Saved center crop to " + output_directory + '/' + name + '-(' + str(initial_x) + ',' + str(initial_y) + ').png'

    for i in range(1, n_patches):

        # what we really need is to get other points within the contour
        neighbors = nearest_neighbors(mask_image, initial_y, initial_x, nn_patch_size)

        count = 0
        while True:
            x = random.choice(np.arange(neighbors[0].size))
            y = random.choice(np.arange(neighbors[0].size))
            if neighbors[y,x] == 255:
                break
            if count >= 500:
                print "[!] Couldn't find a valid patch"
                break
            count += 1

        new_x = keep_in_bounds(initial_x-x)
        new_y = keep_in_bounds(initial_y-y)

        '''
        low_x_value = initial_x - float(patch_size)/patch_frac_denominator # larger denominator: closer to center
        high_x_value = initial_x + float(patch_size)/patch_frac_denominator
        x_translation = np.random.uniform(low=low_x_value, high=high_x_value)
        x_translation = int(x_translation)

        low_y_value = initial_y - float(patch_size)/patch_frac_denominator
        high_y_value = initial_y + float(patch_size)/patch_frac_denominator
        y_translation = np.random.uniform(low=low_y_value, high=high_y_value)
        y_translation = int(y_translation)

        new_x = x_translation # will be shifted off the center
        new_y = y_translation # will be shifted off the center

        new_x = keep_in_bounds(new_x)
        new_y = keep_in_bounds(new_y)
        '''

        shifted_patch = create_patch(image, (new_x, new_y), patch_size)

        cv2.imwrite(output_directory + '/' + name + '-(' + str(new_x) + ',' + str(new_y) + ').png', shifted_patch)
        print "\t\t -> Saved shifted crop to " + output_directory + '/' + name + '-(' + str(new_x) + ',' + str(new_y) + ').png'

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

    mask = image_path[:-4] + '_mask_nuclei.png'
    mask_image = cv2.imread(mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cnts, _ = cv2.findContours(mask_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for point in ground_truth_points:
        try:
            print "\tPositive => " + "(" + str(point[0]) + "," + str(point[1]) + ")"
            # actual mitoses
            gen_patches(image_path, point, mask_image, n_patches = 20, output_directory='training_examples/pos')
            # other points
            for i in range(0, 100): # 100 negative patches
                chosen_contour = random.choice(cnts)
                M = cv2.moments(chosen_contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                tup = (cX, cY)
                print "\tNegative => " + "(" + str(tup[0]) + "," + str(tup[1]) + ")"
                gen_patches(image_path, tup, mask_image, output_directory='training_examples/neg', n_patches=1)
        except Exception as e:
            print e

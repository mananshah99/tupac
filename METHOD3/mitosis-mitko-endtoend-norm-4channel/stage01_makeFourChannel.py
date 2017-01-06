# Performs augmentation on each image in a given directory and saves all augmented images to another directory.
# After this, generates lists of images in the required directory (absolute paths) and ensures that no images are
# in both the train and validation sets.
import numpy as np
import argparse
import math
from random import randint
import os, sys
from scipy.misc import imsave
parser=argparse.ArgumentParser()
parser.add_argument('patch_directory')
parser.add_argument('mask_directory') #for fourth channel
parser.add_argument('output_directory')
parser.add_argument('-r', '--rotate', type=int, help='How many times to randomly rotate any given image', default=4)
parser.add_argument('-a', '--add', type=int, help='Whether to add random values in the range [-15, 15] to each color channel', default=1)
parser.add_argument('-p', '--power', type=int, help='Whether to raise each pixel in any given image to a power in the range [0.95, 1.05]', default=1)

args = parser.parse_args()
n_result = 1 + args.rotate #1 for the original image, args.rotate for the number of rotated copies. Each image will have -a and -p applied to it

# intensity transformation FOR EACH geometrically transformed image
def _internal_augment(image):
    #image.shape => (r, c, 3)

    # same power for the entire image
    P = 1
    
    if args.power == 1:
        P = ((1.05 - 0.95) * np.random.ranf()) + 0.95
    
    for channel in range(0, image.shape[2]):
        
        A = 0

        # add the same number per channel
        if args.add:
            A = np.random.ranf() * 15
            if np.random.ranf() > 0.5:
                A = -A
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                image[x,y,channel] = math.pow(image[x,y,channel], P) + A
    
    return image 

def __augment_help(image, mask):
    # first internal augment
    internal_aug = _internal_augment(image)
    
    # then add rgba channel
    b, g, r = cv2.split(im)
    im_bgra = cv2.merge((b,g,r,mask))


def augment(image, mask):
    # image is in RGB format (np array)
    augmented = []
    augmented.append(image) 
    augmented.append(_internal_augment(image))
    '''
    # screw it, just augment the image and forget rotating
    for i in xrange(args.rotate):
        image_a = image
        #image_a = np.rot90(image, randint(1,3))
        image_a = _internal_augment(image_a)
        augmented.append(image_a)
    '''

    augmented2 = []
    
#    print "Adding fourth channel"
    for im in augmented:
        #print "\t > Splitting image"
        b, g, r = cv2.split(im)
        #print "\t > Merging image"
        im_bgra = cv2.merge((b,g,r,mask))
        augmented2.append(im_bgra)
    
#    print "Done!"
    
    return augmented2

import glob
import cv2
from os.path import join
input_images = glob.glob(args.patch_directory + "/*.png") #these are the input images

if not os.path.exists(args.output_directory):
    print "Creating ", args.output_directory
    os.makedirs(args.output_directory)

def do_work(name):
#for name in input_images:
    if len(glob.glob(args.output_directory + '/' + name.split('/')[-1].split('.')[0] + '*.png')) > 0:
        print "Seen image ", name
        return

    try:
        # patch should look like TUPAC-TR-333_level0_x0000018612_y0000052284.png
        print "Image ", name
        im = cv2.imread(name) # this is the actual patch image, we need the heatmap
        #print "Mask => ", join(args.mask_directory, name.split('/')[-1]) + ".png"
        mask =  cv2.imread(join(args.mask_directory, name.split('/')[-1]) + ".png", cv2.CV_LOAD_IMAGE_GRAYSCALE) # extra .png in mitko pixel heatmaps
        augmented = augment(im, mask)
    
        for idx, i in enumerate(augmented):
            n = args.output_directory + '/' + name.split('/')[-1].split('.')[0] + "-" + str(idx) + ".png"
            imsave(n, i)
    
    except:
        return

from multiprocessing import Pool

p = Pool(20)
p.map(do_work, input_images)

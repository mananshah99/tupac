# Visualizes ALL of the training ROI images and a random sample of other training images 

import cv2
import glob
import sys
from skimage import measure

def get_images_in_dir(directory):
    return glob.glob(directory + "*.png")

def get_csvs_in_dir(directory):
    return glob.glob(directory + "*.csv")

# TUPAC-TR-356-ROI.csv -> TUPAC-TR-356.png
def get_image_name(filename):
    return filename.split("/")[-1].split(".")[0].replace("-ROI", "") + ".png"


def get_true_locations(ground_truth_csv):
    f = open(ground_truth_csv)
    ls = []
    for line in f:
        line = line.strip('\n')
        tmp = line.split(',')
        tmp = [int(int(a)/16.) for a in tmp] # looking at level 2
        ls.append(tmp) #x,y,width,height
    return ls

# directory where original images are
original_image_dir = '/data/dywang/Database/Proliferation/data/TrainingData/small_images-level-2/'

# directory where the ROI masks are
mask_image_dir = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16/'

# directory of the ground truth files
ground_truth_dir = '/data/dywang/Database/Proliferation/data/ROI/ROI/'

# each element of the list is the FULL PATH
original_images = get_images_in_dir(original_image_dir)
mask_images     = get_images_in_dir(mask_image_dir)
ground_truth    = get_csvs_in_dir(ground_truth_dir)

# visualize ALL ground truth ROI images

import time
from tqdm import tqdm

pbar = tqdm(total=len(ground_truth) + 10)

for i, f in enumerate(ground_truth + original_images[0:10]):
    image_name = get_image_name(f)
    
    original_image = cv2.imread(original_image_dir + image_name)
    mask_image     = cv2.imread(mask_image_dir + image_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    _, mask_image   = cv2.threshold(mask_image, int(0.65 * 255), 255, cv2.THRESH_BINARY)

    final_image = cv2.imread(original_image_dir + image_name) #initially set it to the original image
    
    '''
    Assuming (0,0) is @ top left

    x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2
    '''
    for location in get_true_locations(f):
        x1 = location[0]
        y1 = location[1]
        x2 = x1 + location[2]
        y2 = y1 + location[3]
        cv2.rectangle(final_image, (x1, y1), (x2, y2), (0,255, 0), 5)


    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(final_image, contours, -1, (0,0,255), 3)
    
    cv2.imwrite('img/' + image_name, final_image)
    pbar.update(1)
pbar.close()

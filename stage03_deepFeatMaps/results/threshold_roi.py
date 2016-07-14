# RUN THIS, THEN EXTRACT MITOTIC PATCHES

import time
from tqdm import tqdm
import cv2
import numpy as np
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

PATH = 'roi-level1_06-24-16'
threshold_decimal = 0.65

onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

pbar = tqdm(total=len(onlyfiles))

for image in onlyfiles:
    name = image
    #print "On image " + str(image)
    image = cv2.imread(join(PATH, image), cv2.CV_LOAD_IMAGE_GRAYSCALE)
    _, thresholded = cv2.threshold(image, int(255 * threshold_decimal), 255, cv2.THRESH_BINARY)
    cv2.imwrite(join(str(PATH + '/thresholded-' + str(threshold_decimal)), name), thresholded)
    pbar.update(1)
pbar.close()

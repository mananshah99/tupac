import cv2
import numpy as np

NUM ='493'

tissue_mask = '/data/dywang/Database/Proliferation/data/TrainingData/small_images-level-2-mask/TUPAC-TR-' + NUM + '.png'
roi_mask    = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16/TUPAC-TR-' + NUM + '.png'

tissue = cv2.imread(tissue_mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)
roi    = cv2.imread(roi_mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Number of white pixels in each image
white_tissue = cv2.countNonZero(tissue)
white_roi    = cv2.countNonZero(roi)

overall = tissue.size

print "Tissue white percentage \t" + str(float(white_tissue) / overall)
print "ROI white percentage \t" + str(float(white_roi) / overall)
print "ROI reduction over tissue \t" + str(float(white_roi - white_tissue) / white_tissue)

import cv2
import numpy as np

#NUM ='003'

def diff(NUM):
    tissue_mask = '/data/dywang/Database/Proliferation/data/TrainingData/small_images-level-2-mask/TUPAC-TR-' + NUM + '.png'
    roi_mask    = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16/TUPAC-TR-' + NUM + '.png'

    tissue = cv2.imread(tissue_mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    roi    = cv2.imread(roi_mask, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    # Number of white pixels in each image
    white_tissue = cv2.countNonZero(tissue)
    white_roi    = cv2.countNonZero(roi)

    overall = tissue.size

    #print "Tissue white percentage \t" + str(float(white_tissue) / overall)
    #print "ROI white percentage \t" + str(float(white_roi) / overall)
    #print "ROI reduction over tissue \t" + str(float(white_roi - white_tissue) / white_tissue)
    return float(white_roi - white_tissue) / white_tissue

from tqdm import tqdm

a = [str(i).zfill(3) for i in range(0, 500)]
d_ave = []
for i in tqdm(a):
    try:
        d = diff(i)
        d_ave.append(d)
    except:
        continue

print d_ave
print np.average(np.array(d_ave))

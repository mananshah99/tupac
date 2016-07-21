from __future__ import print_function, division

import openslide as os
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2hed, rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu
from skimage import img_as_int
from multiprocessing import Pool

import sys
## FIX THIS SOON
sys.path.append("Train")
import tools

DESCRIPTION = """
Name: stage01-2_findTissueRegion.py
    => Stage 1, step 2 of the pipeline

Description: finds the tissue regions in each SMALL segment
from a whole slide image (generated by stage01-1). 

Dependencies: 
    * OpenSlide
    * skimage
	* numpy
	* matplotlib
	* multiprocessing
    * OpenCV (cv2)
    * tools.py in the same directory
"""

'''
doFindROI

finds the region of interest in a given image using the function
tools.findROI, which utilizes OTSU thresholding & morphological
operations.

Note that, here, the ROI is not a small rectangular region of interest
from which to extract specfic mitoses; rather, it is a mask of the tissue
region. 

'''
def doFindROI(imgName, saveName):
    img = imread(imgName)
    msk = tools.findROI(img)
    imsave(saveName, msk)

'''
run_findROI_tumor

'''
def run_findROI_tumor(ID):
    print(ID)
    imgName = 'tr_t/Tumor_%03d.png'%(ID)
    mskName1 ='tr_tm/Tumor_%03d.png'%(ID)
    doFindROI(imgName, mskName1)

    mskName2 ='tr_t_m/Tumor_%03d_Mask.png'%(ID)
    outputName ='tr_t_overlay/Tumor_%03d_overlay.png'%(ID)
    tools.doAddMasks(imgName,
                     [mskName1, mskName2],
                     [[0, 255, 0],[0, 0, 255]],
                     outputName)

def run_findROI_normal(ID):
    print(ID)
    imgName = 'tr_n/Normal_%03d.png'%(ID)
    mskName1 ='tr_nm/Normal_%03d.png'%(ID)
    doFindROI(imgName, mskName1)
    outputName ='tr_n_overlay/Normal_%03d_overlay.png'%(ID)
    tools.doAddMask(imgName,
                     mskName1,
                     [0, 255, 0],
                     outputName)

def run_findROI_test(ID):
    print(ID)
    imgName = 'TestsetImg/Test_%03d.png'%(ID)
    mskName1 ='TestsetMsk/Test_%03d.png'%(ID)
    doFindROI(imgName, mskName1)
    outputName ='TestsetImgOverlay/Test_%03d_overlay.png'%(ID)
    tools.doAddMask(imgName,
                     mskName1,
                     [0, 255, 0],
                     outputName, msktype = 1)
                     
def check_ROI_OK(ID):
    mskName1 ='tr_tm/Tumor_%03d.png'%(ID)
    mskName2 ='tr_t_m/Tumor_%03d_Mask.png'%(ID)
    
    msk1 = imread(mskName1, True)
    msk2 = imread(mskName2, True)
    msk = msk1 - msk2
    v = np.sum(msk<0)
    if v == 0:
        return True
    else:
        return False

def run_findROI(ID):
    print("On ID " + str(ID))
    
    try: 
        imgName = '../../data/TrainingData/small_images-level-2/TUPAC-TR-%03d.png'%(ID)	
        mskName1 ='../../data/TrainingData/small_images-level-2-mask/TUPAC-TR-%03d.png'%(ID)
    
        doFindROI(imgName, mskName1)

        outputName = '../../data/TrainingData/small_images-level-2-overlay/TUPAC-TR-%03d.png'%(ID)

        tools.doAddMask(imgName,
                     mskName1,
                     [0, 255, 0],
                     outputName)
    except:
        print("[!] ID " + str(ID) + " failed.")
        pass

if __name__ == "__main__":

	# remember to modify the level in run_findROI
	# generates a multiprocessing pool
    pool = Pool(100)
	
    ID1, ID2 = 1,500
    IDs = range(ID1, ID2 + 1)
   
    pool.map(run_findROI, IDs)
    
    #for ID in IDs:
    #    try: 
    #        run_findROI(ID)
    #    except:
    #        print "Finding ROI failed for ID " + str(ID)

    # pool.map(run_findROI, IDs)
    #for ID in IDs:
    #    if not check_ROI_OK(ID):
    #        print("ROI is too small for image ", ID)


	#===there is no normal/tumor distinction in these images===

	#ID1, ID2 = 1, 110
    #IDs =range(ID1, ID2+1)
    #pool.map(run_findROI_tumor, IDs)
    #for ID in IDs:
    #    if not check_ROI_OK(ID):
    #        print("Too small ROI",ID)
    #ID1, ID2 = 1, 160
    #IDs =range(ID1, ID2+1)
    #pool.map(run_findROI_normal, IDs)
    
    #ID1, ID2 = 1, 130
    #IDs =range(ID1, ID2+1)
    #pool.map(run_findROI_test, IDs)
#   run_findROI_test(IDs[0])

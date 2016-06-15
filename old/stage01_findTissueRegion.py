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
sys.path.append("Train")
import tools

def doFindROI(imgName, saveName):
    img = imread(imgName)
    msk = tools.findROI(img)
    imsave(saveName, msk)

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
    msk1 = imread(mskName1,True)
    msk2 = imread(mskName2,True)
    msk = msk1 - msk2
    v = np.sum(msk<0)
    if v == 0:
        return True
    else:
        return False

if __name__ == "__main__":
    pool = Pool(10)
    #ID1, ID2 = 1, 110
    #IDs =range(ID1, ID2+1)
    #pool.map(run_findROI_tumor, IDs)
    #for ID in IDs:
    #    if not check_ROI_OK(ID):
    #        print("Too small ROI",ID)
    #ID1, ID2 = 1, 160
    #IDs =range(ID1, ID2+1)
    #pool.map(run_findROI_normal, IDs)
    
    ID1, ID2 = 1, 130
    IDs =range(ID1, ID2+1)
    pool.map(run_findROI_test, IDs)
#    run_findROI_test(IDs[0])

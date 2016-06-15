from __future__ import print_function, division

import openslide as osi
from skimage.io import imread, imsave
import numpy as np
import cv2
import sys
sys.path.append("..")
import tools

DESCRIPTION = """
Name: stage01-1_genSmallImage.py
	=> Stage 1, step 1 of the pipeline

Description: generates small images from whole slide images.

Dependencies: 
	* OpenSlide
	* skimage
	* OpenCV (cv2)
	* tools.py in the same directory
"""

# Sample image name: TUPAC-TR-001.svs

'''
getSmallImages

iDIR: input directory (to load the whole slide image)
oDIR: output directory (to save the small image)
prefix: identifier name (e.g. Tumor, Normal, Mask)
appendix: any ending string (generally just '')
LEVEL: magnification level of the .svs image
IDs: the singular ID identifying the initial image

Note: the whole slide image name is defined as
	[iDIR]/[prefix]_[IDs][appendix].svs
'''
def getSmallImages(iDIR, oDIR, prefix, appendix, LEVEL, IDs):
    def getSmallImage(wsi_name, output_name, LEVEL):
        print("Whole slide image name \t %s" % wsi_name)
        wsi = osi.OpenSlide(wsi_name)
        print("\t Image dimensions @ level 0 \t", wsi.dimensions)
        dim = wsi.level_dimensions[LEVEL]
        print("\t Image dimensions @ level " + str(LEVEL) + "\t", dim)
        img = wsi.get_thumbnail(dim)
        print("\t RGB image dimensions (width, height) ", img.width, img.height)
        imsave(output_name, np.asarray(img))
	
	# iDIR		../data/TrainingData/training_image_data/
	# prefix	TUPAC-TR-
	# IDs		[001 - ??]
	# appendix	'' (none)
    
	# (for reference)  wsi_name = '%s/%s_%03d%s.svs'%(iDIR, prefix, IDs, appendix)
    wsi_name = '%s/%s%03d%s.svs'%(iDIR, prefix, IDs, appendix)
	# (for reference)  output_name ='%s/%s_%03d%s.png'%(oDIR, prefix, IDs, appendix)
    output_name ='%s/%s%03d%s.png'%(oDIR, prefix, IDs, appendix)
	
    getSmallImage(wsi_name, output_name, LEVEL)

# For later, marked @ June 15, 2016
def addMasks(IDs):
    print(IDs)
    imgName = 'tr_t/Tumor_%03d.png'%(IDs)
    mskName = 'tr_t_m/Tumor_%03d_Mask.png'%(IDs)
    outputName = 'tr_t_ma/Tumor_%03d.png'%(IDs)
    tools.doAddMask(imgName, mskName, [0, 255, 0], outputName)

if __name__ == "__main__":
    ID1, ID2, LEVEL = 1, 500, 2 #[0,4] but try 0 first
    [getSmallImages('../data/TrainingData/training_image_data/', '../data/TrainingData/small_images-level-' + str(LEVEL) + '/', "TUPAC-TR-", "", LEVEL, IDs) for IDs in range(467,468)] #range(ID1, ID2+1)]

    #ID1, ID2, LEVEL = 1, 110, 2
    #[getSmallImages('../Train_Tumor', 'Tumor', "Tumor", "", LEVEL, IDs) for IDs in range(ID1, ID2+1)]

    #ID1, ID2, LEVEL = 1, 110, 5
    #[getSmallImages("Mask", "tr_t_m", "Tumor",  "_Mask", LEVEL, IDs) for IDs in range(ID1, ID2+1)]

    #ID1, ID2, LEVEL = 1, 70, 5
    #[addMasks(IDs) for IDs in range(ID1, ID2+1)]

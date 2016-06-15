from __future__ import print_function, division

import openslide as osi
from skimage.io import imread, imsave
import numpy as np
import cv2
import sys
sys.path.append("..")
import tools

def getSmallImages(iDIR, oDIR, prefix, appendix, LEVEL, IDs):
    def getSmallImage(wsi_name, output_name, LEVEL):
        print("%s..."%wsi_name)
        wsi = osi.OpenSlide(wsi_name)
        print("wsi: ", wsi.dimensions)
        dim = wsi.level_dimensions[LEVEL]
        print("dim: ", dim)
        img = wsi.get_thumbnail(dim)
        print("img: ", img.width, img.height)
        imsave(output_name, np.asarray(img))

    wsi_name = '%s/%s_%03d%s.tif'%(iDIR, prefix, IDs, appendix)
    output_name ='%s/%s_%03d%s.png'%(oDIR, prefix, IDs, appendix)
    getSmallImage(wsi_name, output_name, LEVEL)

def addMasks(IDs):
    print(IDs)
    imgName = 'tr_t/Tumor_%03d.png'%(IDs)
    mskName = 'tr_t_m/Tumor_%03d_Mask.png'%(IDs)
    outputName = 'tr_t_ma/Tumor_%03d.png'%(IDs)
    tools.doAddMask(imgName, mskName, [0, 255, 0], outputName)

if __name__ == "__main__":
    ID1, ID2, LEVEL = 1, 160, 2
    [getSmallImages('../Train_Normal', 'Normal', "Normal", "", LEVEL, IDs) for IDs in range(ID1, ID2+1)]

    ID1, ID2, LEVEL = 1, 110, 2
    [getSmallImages('../Train_Tumor', 'Tumor', "Tumor", "", LEVEL, IDs) for IDs in range(ID1, ID2+1)]

    #ID1, ID2, LEVEL = 1, 110, 5
    #[getSmallImages("Mask", "tr_t_m", "Tumor",  "_Mask", LEVEL, IDs) for IDs in range(ID1, ID2+1)]

    #ID1, ID2, LEVEL = 1, 70, 5
    #[addMasks(IDs) for IDs in range(ID1, ID2+1)]

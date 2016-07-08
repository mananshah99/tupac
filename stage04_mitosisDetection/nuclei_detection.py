# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:32:05 2016

@author: dayong
"""
import numpy as np
from scipy import ndimage as nd
import skimage as ski
import skimage.io as skio
from skimage.exposure import rescale_intensity
from skimage import morphology
import scipy.ndimage.morphology as smorphology
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from matplotlib import pylab as plt
from skimage.segmentation import clear_border
#from skimage.filters import gaussian_filter

def nuclei_detection(img, MinPixel, MaxPixel):
    img_f = ski.img_as_float(img)
    adjustRed = rescale_intensity(img_f[:,:,0])
    roiGamma = rescale_intensity(adjustRed, in_range=(0, 0.5));
    roiMaskThresh = roiGamma < (250 / 255.0) ;

    roiMaskFill = morphology.remove_small_objects(~roiMaskThresh, MinPixel);
    roiMaskNoiseRem = morphology.remove_small_objects(~roiMaskFill,150);
    roiMaskDilat = morphology.dilation(roiMaskNoiseRem, morphology.disk(3));
    roiMask = smorphology.binary_fill_holes(roiMaskDilat)

    hsv = ski.color.rgb2hsv(img);
    hsv[:,:,2] = 0.8;
    img2 = ski.color.hsv2rgb(hsv)
    diffRGB = img2-img_f
    adjRGB = np.zeros(diffRGB.shape)
    adjRGB[:,:,0] = rescale_intensity(diffRGB[:,:,0],in_range=(0, 0.4))
    adjRGB[:,:,1] = rescale_intensity(diffRGB[:,:,1],in_range=(0, 0.4))
    adjRGB[:,:,2] = rescale_intensity(diffRGB[:,:,2],in_range=(0, 0.4))

    gauss = gaussian_filter(adjRGB[:,:,2], sigma=3, truncate=5.0);

    bw1 = gauss>(100/255.0);
    bw1 = bw1 * roiMask;
    bw1_bwareaopen = morphology.remove_small_objects(bw1, MinPixel)
    bw2 = smorphology.binary_fill_holes(bw1_bwareaopen);

    bwDist = nd.distance_transform_edt(bw2);
    filtDist = gaussian_filter(bwDist,sigma=5, truncate=5.0);

    L = label(bw2)
    R = regionprops(L)
    coutn = 0
    for idx, R_i in enumerate(R):
        if R_i.area < MaxPixel and R_i.area > MinPixel:
            r, l = R_i.centroid
            #print(idx, filtDist[r,l])
        else:
            L[L==(idx+1)] = 0
    BW = L > 0
    return BW

'''
if __name__ == "__main__":
    MinPixel = 200
    MaxPixel = 2500
    #img_name = 'Normal_085_0000000002_0000000001.png'
    #img_name = 'Tumor_008_0000000006.png'
    img_name = '/home/dayong/ServerDrive/GPU/data/Database/MITOS/A/test/A00_00.bmp'


    img = skio.imread(img_name)
    #sn = stain_normliazation('/home/dayong/ref.png')
    #nimg = sn.stain(img)
    bw1 = nuclei_detection(img, MinPixel, MaxPixel)
    #bw2 = nuclei_detection(nimg, MinPixel, MaxPixel)
    bw3 = nuclei_detection_cancer(img, MinPixel, MaxPixel)
    #bw4 = nuclei_detection_cancer(nimg, MinPixel, MaxPixel)
    plt.figure()
    plt.subplot(141); skio.imshow(bw1)
    #plt.subplot(142); skio.imshow(bw2)
    plt.subplot(143); skio.imshow(label(bw3))
    #plt.subplot(144); skio.imshow(label(bw4))

    #skio.imsave('a1.png', bw3 * 255)
    #skio.imsave('a2.png', bw4 * 255)
    plt.show()
'''

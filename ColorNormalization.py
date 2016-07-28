import os
import cv2
import sys
import openslide as osi
import skimage.io as skio
import numpy as np
from random import sample
from skimage.transform import resize
from scipy import sparse, io
import scipy.ndimage as nd
import spams
import copy
import time
import math
from os.path import join
from random import randint
import random
from sklearn.feature_extraction.image import extract_patches_2d
from multiprocessing import Pool
from tqdm import tqdm

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class ColorNormalization:
    
    VERBOSITY = 0     # default is 0    
    wsi = None        # the entire image
    mask = None       # at level 2 (roi mask)    
    smallwsi = None   # at level 2 
    patch_coords = None # patch coordinates

    levelpow = 4

    intensity_max = 240
    opticalDensity_minThreshold = 0.15 

    DEFAULT_HE_REF_0 = [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]    # Reference H&E stain vecotrs matrix (respective column vectors)
    DEFAULT_HE_REF_0_MAX_CONCENTRATION = [1.9705, 1.0308]                        # Reference maximum H&E stain concentrations

    HE_ref = np.array(DEFAULT_HE_REF_0, dtype=np.float)
    HE_maxConcentration_ref = np.array(DEFAULT_HE_REF_0_MAX_CONCENTRATION, dtype=np.float)
    globalStainMatrix = None

    outmap = None

    def extend_inds_to_level0(self, input_level, h, w):
        gap = input_level - 0
        v = np.power(self.levelpow, gap)
        hlist = h * v + np.arange(v)
        wlist = w * v + np.arange(v)
        hw = []
        for hv in hlist:
            for wv in wlist:
                hw.append([hv, wv])
        return hw

    def get_tl_pts_in_level0(self, OUT_LEVEL, h_level0, w_level0, wsize):
        scale = np.power(self.levelpow, OUT_LEVEL)
        wsize_level0 = wsize * scale
        wsize_level0_half = wsize_level0 / 2

        h1_level0, w1_level0 = h_level0 - wsize_level0_half, w_level0 - wsize_level0_half
        return int(h1_level0), int(w1_level0)

    def get_image(self, wsi, h1_level0, w1_level0, OUT_LEVEL, wsize):
        img = wsi.read_region(
                  (w1_level0, h1_level0),
                  OUT_LEVEL, (wsize, wsize))
        img = np.asarray(img)[:,:,:3]
        return img

    # it is initialized using a whole slide image and a tissue mask
    # and a small whole slide image @ level 2 (for getting patches)
    def __init__(self, img_name, msk_name, small_img_name = None, verbosity=0):
        self.VERBOSITY = verbosity
        print "I am on verbose level ", self.VERBOSITY
        self.wsi = osi.OpenSlide(img_name)
        self.mask = cv2.imread(msk_name, cv2.IMREAD_GRAYSCALE)

        if small_img_name is None:
            number = img_name.split('TUPAC-TR-')[1]
            small_img_name = img_name.split('training_image_data')[0] #(everything before /training_image_data/)
            small_img_name = join(small_img_name, 'small_images-level-2-mask/TUPAC-TR-' + number[:-4] + '.png')

        self.smallwsi = cv2.imread(small_img_name) # this is bgr

        if self.VERBOSITY == 1:
            print "\t __init__ : loaded ", img_name
            print "\t __init__ : loaded ", msk_name
            print "\t __init__ : loaded ", small_img_name        
        self.train()
        self.get_normalizaed_wsi('a')


    def _1Dto2D(self, image): #-- Input 1D array of rgb vectors transformed to 2D array of pixels
        if len(image.shape) > 1:
            depth = image.shape[0]
            image_2d = np.zeros((1000, 1000, 3), dtype=np.int)
            for k in range(depth):
                temp = np.reshape(image[k,:], (1000, 1000), order='F')
                image_2d[:,:,k] = temp
        else:
            image_2d = np.reshape(image, (1000, 1000), order='F')

        return image_2d

    def _od2rgb(self, image): #-- Convert input image from OD to RGB space
        return self.intensity_max * np.exp(-1* image) - np.ones(image.shape, dtype=image.dtype)

    def _rgb2od(self, image): #-- Convert input image from RGB to OD space
        image = np.divide(self.intensity_max*np.ones(image.shape, dtype=image.dtype), image + np.ones(image.shape, dtype=image.dtype))
        return np.log(image)

    def _concentrationContrastNormalize(self, concentrations, maxC=None):
        """
        Normalize concentrations contrast according to a regular numpy array representation
        """
        for k in range(2): #-- For each stain
            c_max = 0
        if maxC:
            c_max = maxC[k]
        else:
            c_max = np.amax(concentrations[k,:])
            concentrations[k,:] *= self.HE_maxConcentration_ref[k] / c_max #-- Scale concentration range to get the same reference maximum
        return concentrations

    def _2Dto1D(self, image): #-- Input 2D array of pixels transformed as a 1D array of rgb vectors
        rows, cols, depth = image.shape
        image = image.transpose()
        image = image.reshape((depth, rows*cols))
        return image

    def _odThresholding(self, image): #-- Threshold transparent pixels with defined threshold
        validPixels_mask = image > self.opticalDensity_minThreshold
        validPixels_mask = validPixels_mask[0,:] * validPixels_mask[1,:] * validPixels_mask[2,:]
        return image[:, validPixels_mask]

    def _stainMatrixOrdering(self, matrix): #-- Reorder arbitrary matrix to expected H-E order
        if matrix[0,0] > matrix[0,1]: #-- First colum blocks more red => it is likely to be the color of H
            return matrix
        else:
            return matrix[:,[1,0]] #-- Swao column vectors

    # input: an image patch
    # output: the normalized image
    # this part is related to the normalization method
    def convert(self, img): #img MUST be rgb => this returns rgb
        I_rgb = self._2Dto1D(img)
        I_od = self._rgb2od(I_rgb)

        concentrations = np.linalg.pinv(self.globalStainMatrix).dot(I_od)
        concentrations = np.multiply(concentrations>0, concentrations)

        beta = 0.99
        #-- Reference scaling of concentration ranges
        sorted_idx = np.argsort(concentrations, axis=1) #-- Sorted concentrations
        idx_max = int(beta*sorted_idx.shape[1])
        maxC = [concentrations[k, idx] for k,idx in enumerate(sorted_idx[:,idx_max])] #-- maximum concentration (beta-percentile)

        #-- alpha-th maximum angles
        concentrations = self._concentrationContrastNormalize(concentrations, maxC)

        #-- Normalize stain vectors by changing vector base
        I_od_norm = self.HE_ref.dot(concentrations)

        #-- Conversion back to RGB space
        I_rgb_norm = self._od2rgb(I_od_norm)
        #print "Normalized RGB:", I_rgb_norm[0,0:20]

        I_rgb_norm = np.array(I_rgb_norm, dtype=np.uint8) #-- Uint8 type conversion
        I_rgb_norm = self._1Dto2D(I_rgb_norm) #-- 2D conversion

        return I_rgb_norm 

    # train the color normalization model
    # randomly collected several patches from the tissue regions, and
    # compute the color normalization model
    # this part is related to the normalization method
    def train(self, n_random=5, patch_size = 1000):
        patches = []
        if self.VERBOSITY == 1:
            print "\t train : initializing " + str(n_random) + " patches"
        
        indices = zip(*np.where(self.mask == 255))
        n = 0
        height, width = self.mask.shape
        while n < n_random:

            h, w = random.choice(indices)

            # top left
            #h = randint(0, height - patch_size)
            #w = randint(0, width  - patch_size)

            #print self.mask[h:h+patch_size, w:w+patch_size]
                
            #if any(0 in sublist for sublist in self.mask[h:h+patch_size, w:w+patch_size]):
            #    continue  
        
            indices = self.extend_inds_to_level0(2, h, w)
            
            idx = int(len(indices) / 2)
            chcw_level0 = indices[idx] # take middle point (this is basically the centroid at level 0)
            h_level0, w_level0 = chcw_level0
            h1_level0, w1_level0 = self.get_tl_pts_in_level0(0,
                                                        h_level0,
                                                        w_level0,
                                                        patch_size) # should give top left corner of patch

            patch = self.get_image(self.wsi, h1_level0, w1_level0, 0, patch_size)
            if self.VERBOSITY == 1:
                print "\t train :  => obtained patch with coordinates ", h1_level0, w1_level0
            
            patches.append(patch)
            n += 1

        start = time.time()
        if self.VERBOSITY == 1:
            print "\t train : reshaping images to 1D"
        
        out = np.empty((3, n_random * 1000000))
        for idx, p in enumerate(patches):
            p1d = self._2Dto1D(p)
            out[:,idx * 1000000 : (idx + 1)*1000000] = p1d
       
        if self.VERBOSITY == 1:
            print "\t train : running _rgb2od and _odThresholding"
  
        I_od  = self._rgb2od(out) 
        I_odT = self._odThresholding(I_od)    

        if self.VERBOSITY == 1:
            print "\t train : performing eigen analysis"        
        
        #-- Calculate eigenvectors
        I_od_cov = np.cov(I_odT) #-- Covariance matrix
        I_eigVal, I_eigVect = np.linalg.eigh(I_od_cov) #-- Eigen analysis
        I_eigVect = np.multiply(I_eigVect, 1-2*np.all(I_eigVect <=0, axis=0)) #-- Potential sign swapping
        sorted_idx = np.argsort(I_eigVal)[::-1] #-- Reversed sorted list
        
        if self.VERBOSITY == 1:
            print "\t train : performing project matrix computation"
        
        projection_matrix = np.array([I_eigVect[:,idx] for idx in sorted_idx[0:2]], dtype=np.float) #-- Projection matrix
        I_od_proj = projection_matrix.dot(I_odT) #-- Projection weights        

        if self.VERBOSITY == 1:
            print "\t train : computing angle list"

        alpha = 0.01        
        phi = np.arctan2(I_od_proj[0,:], I_od_proj[1,:]) #-- Compute all the angles
        sorted_idx = np.argsort(phi)
        idx_min, idx_max = int((1.0-alpha)*len(sorted_idx)), int(alpha*len(sorted_idx))
        phi_min, phi_max = phi[sorted_idx[idx_min]], phi[sorted_idx[idx_max]] #-- alpha-th lowest and alpha-th highest angles
        #-- Stain vector estimations
        stainMatrix_proj = np.array([[np.sin(phi_min), np.sin(phi_max)], [np.cos(phi_min), np.cos(phi_max)]], dtype=np.float) #-- Matrix of the estimated stain vectors in the fit plane
        stainMatrix = np.transpose(projection_matrix).dot(stainMatrix_proj) #-- Projection back to OD space
        stainMatrix = self._stainMatrixOrdering(stainMatrix)
        
        print "\t train : stain matrix shape ", stainMatrix.shape

        self.globalStainMatrix = stainMatrix
        
        print "\t train : took ", time.time() - start

    def process_patch(self, coordinate): #coordinate is a tuple
        try:
            patch = self.get_image(self.wsi, coordinate[0], coordinate[1], 0, 1000)
            if (all(x == 0 for x in patch)):
                print "\t process_patch : skipped due to all 0"
                return #outmap is initialized to zeros
            patch = self.convert(patch)
            #cv2.imwrite('testdir/converted-' + str(coordinate[0]) + '=' + str(coordinate[1]) + '.png', patch[:,:,(2,1,0)])
            self.outmap[coordinate[0] : coordinate[0] + patch.shape[0], coordinate[1] : coordinate[1] + patch.shape[1]] = patch
            print "\t process_patch : wrote coordinates ", coordinate
        except Exception as e:
            print e
            print e.args
            #print patch.shape
            #print self.outmap[coordinate[0] : coordinate[0] + patch.shape[0], coordinate[1] : coordinate[1] + patch.shape[1]].shape

    # scan the whole slide image, and convert the patches, independently
    # this part is independent of the normalization method
    def get_normalizaed_wsi(self, result_name):
        ## step 01 get all the image patches locations @ level 0
        ## a list of (left_top_x, left_top_y, width, height)
       
        dimensions = self.wsi.dimensions
        mapf = 'memmap.dat'
        self.outmap = np.memmap(mapf, dtype=int, mode='w+', shape=(dimensions[0], dimensions[1], 3))
        
        if self.VERBOSITY == 1:
            print "\t get_normalized_wsi : dimensions are ", dimensions, " -> ", mapf

        coordinates = []
        for i in xrange(0, dimensions[0], 1000):
            for j in xrange(0, dimensions[1], 1000):
                coordinates.append((i,j))

        if self.VERBOSITY == 1:
            print "\t get_normalized_wsi : there are ", len(coordinates), " patches"

        self.patch_coordinates = coordinates

#        bar = tqdm(total=len(coordinates))
#        for c in coordinates:
#            self.process_patch(c) 
#            bar.update(1)
#        bar.close()
#        print self.outmap
           
if __name__ == "__main__":
    cn = ColorNormalization(img_name, msk_name)
    #cn.get_normalizaed_wsi(result_name)

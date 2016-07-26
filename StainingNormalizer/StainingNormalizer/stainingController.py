# -*- coding: utf-8 -*-
"""
Created in May 2016
@author: mlafarge

References:
    MACENKO'S METHOD:
    A method for normalizing histology slides for quantitative analysis. M.
    Macenko et al., ISBI 2009

    Efficient nucleus detector in histopathology images. J.P. Vink et al., J
    Microscopy, 2013

    Matlab implementation by Mitko Veta, 2013
    https://github.com/mitkovetta/staining-normalization

    VAHADANE'S METHOD:

"""

import numpy as np
import copy

import spams        # SPAMS library
import constants    # Keys and parameters


class Controller:
    """
    Class grouping different staining normalization algorithms
    The initializer associates keys to related methods
    Desired methods can be then applied with +compute
    """
    #- Constants of the class
    DTYPE = np.float
    def __init__(self, image):
        """
        Class initializer
        INPUT:
            - image: RGB input image
        """
        #-
        #- Definition of attributes of the instance which are in common for all the methods
        #- Those attribute can be user-tuned
        #-
        self._image = None # Input image
        self._rows, self._cols, self._depth = 0, 0, 0

        self.intensity_max = constants.DEFAULT_INTENSITY_MAX
        self.opticalDensity_minThreshold = constants.DEFAULT_OPTICAL_DENSITY_MIN_THRESHOLD
        self.HE_ref = np.array(constants.DEFAULT_HE_REF_0, dtype=self.DTYPE)
        self.HE_maxConcentration_ref = np.array(constants.DEFAULT_HE_REF_0_MAX_CONCENTRATION, dtype=self.DTYPE)

        self.spams_dl_legal = constants.DEFAULT_SPAMS_DL_LEGAL
        self.spams_lasso_legal = constants.DEFAULT_SPAMS_LASSO_LEGAL

        #-
        #- Inputs consideration
        #-
        self.setImage(image)

    def setImage(self, image):
        self._image = np.array(image, dtype=self.DTYPE) #-- Copy of the input image
        self._rows, self._cols, self._depth = image.shape

        return 1

    def __call__(self, function, parameters={}):
        return function(self, **parameters)

    def compute(self, methodKey, setParams={}):
        if methodKey in self._parameters and methodKey in self._methods:
            #-- Update method-related parameters
            self.setParameters(methodKey, setParams)
            return self._methods[methodKey]() #-- Execution of the input method
        return 0

    #--
    #-- General methods
    #--
    def _2Dto1D(self, image): #-- Input 2D array of pixels transformed as a 1D array of rgb vectors
        rows, cols, depth = image.shape
        image = image.transpose()
        image = image.reshape((depth, rows*cols))
        return image

    def _1Dto2D(self, image): #-- Input 1D array of rgb vectors transformed to 2D array of pixels
        if len(image.shape) > 1:
            depth = image.shape[0]
            image_2d = np.zeros((self._rows, self._cols, depth), dtype=image.dtype)
            for k in range(depth):
                temp = np.reshape(image[k,:], (self._rows, self._cols), order='F')
                image_2d[:,:,k] = temp
        else:
            image_2d = np.reshape(image, (self._rows, self._cols), order='F')

        return image_2d
        """
        image = image.reshape((image.shape[1], self._cols, self._rows), order='F')
        image = image.reshape((self._cols, self._rows), order='F')
        image = image.transpose()
        return image
        """

    def _rgb2od(self, image): #-- Convert input image from RGB to OD space
        image = np.divide(self.intensity_max*np.ones(image.shape, dtype=image.dtype), image + np.ones(image.shape, dtype=image.dtype))
        return np.log(image)

    def _od2rgb(self, image): #-- Convert input image from OD to RGB space
        return self.intensity_max * np.exp(-1* image) - np.ones(image.shape, dtype=image.dtype)

    def _odThresholding(self, image): #-- Threshold transparent pixels with defined threshold
        validPixels_mask = image > self.opticalDensity_minThreshold
        validPixels_mask = validPixels_mask[0,:] * validPixels_mask[1,:] * validPixels_mask[2,:]
        return image[:, validPixels_mask]

    def _stainMatrixOrdering(self, matrix): #-- Reorder arbitrary matrix to expected H-E order
        if matrix[0,0] > matrix[0,1]: #-- First colum blocks more red => it is likely to be the color of H
            return matrix
        else:
            return matrix[:,[1,0]] #-- Swao column vectors

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

    def _spams_reshapeSparseEstimation(self, matrix):
        """
        Reshape the input concentrations matrix
        convert the specific data structure of the output of spams.lasso (sparse representation as a list of pointer to non-negative entries)
        """
        matrix_new = np.zeros(matrix.shape, dtype=matrix.dtype) #-- New concentrations numpy array
        nb_updates = len(matrix.data) #-- Number of values which need to be updated

        nbPx = matrix.shape[1]
        keys_depth = [0]*nb_updates #-- List of depths with data
        keys_px = [0]*nb_updates #-- List of related pixel indexes

        for px in range(nbPx): #-- Scan of all the pointers
            curr_ptr = matrix.indptr[px] #-- Pointer of the current pixel
            last_depth = -1
            while (curr_ptr < nb_updates and matrix.indices[curr_ptr] > last_depth):
                keys_px[curr_ptr] = px #-- Px id
                last_depth = matrix.indices[curr_ptr]
                curr_ptr += 1

        matrix_new[matrix.indices, keys_px] = matrix.data #-- Fast value replacement
        return matrix_new





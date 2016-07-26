# -*- coding: utf-8 -*-
"""
Created in May 2016
@author: mlafarge

Python implementation of Vahadane's method for staining normalization 2016

References:
    VAHADANE'S METHOD:
    Vahadane, Abhishek, et al.
    "Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images." (2016).

    SPARSE MODELING SOFTWARE:
    J. Mairal, F. Bach, J. Ponce and G. Sapiro.
    Online Learning for Matrix Factorization and Sparse Coding. Journal of Machine Learning Research, volume 11, pages 19-60. 2010.

    J. Mairal, F. Bach, J. Ponce and G. Sapiro.
    Online Dictionary Learning for Sparse Coding. International Conference on Machine Learning, Montreal, Canada, 2009
"""

import copy
import numpy as np
import spams

DEFAULT_PARAMETERS = {
    "spamsParameters": { #-- Dictionary of parameters required for sparse dictionary learning (SPAMS library)
        "K": 2,                 #-- learns a dictionary of 2 atoms (2 stain vectors)
        "return_model": False,
        "posAlpha": True,       #-- Positivity Constraint on the decomposition weights
        "pos": True,            #-- Positivity Constraint on atom values
        "lambda1": 0.1,         #-- Regularization constant
        "batchsize": 256**2,    #-- Size of the random batch selected to train at each iteration
        "iter": 50,             #-- Number of iterations
        "ols": False            #-- Final orthogonal projection for Lasso minimization
    }
}

def vahadane(controller,
    spamsParameters = DEFAULT_PARAMETERS["spamsParameters"]): #-- Parameters for SPAMS library
    """
    Vahadane's method for staining normalization
    INPUT:
        (StainingController) instance for global normalization parameter state
    OUTPUT:
        (numpy array) I_rgb_norm:     normalized image
        (numpy array) stainMatrix:    unmixing matrix (stain vectors as columns)
        (numpy array) concentrations: concentration maps
        (numpy array) I_od_noise:     map of the noise in OD space
    """

    I_rgb = copy.deepcopy(controller._image) #-- Copy of the original image

    #-- Image reshaping: list of pixel vectors
    I_rgb = controller._2Dto1D(I_rgb)

    #-- Optical Density convertion
    I_od= controller._rgb2od(I_rgb)

    #-- Transparent pixels thresholding
    I_odT = controller._odThresholding(I_od)

    #-- Sparse dictionary learning (dictionary of stain vectors)
    #-- Parameters filtering for dictionary learning
    currentParameters = {}
    currentParameters.update([(key, spamsParameters[key]) for key in controller.spams_dl_legal if key in spamsParameters])
    dictionary = spams.trainDL(I_odT, **currentParameters) #-- Learns dictionary of stain vectors (as column vectors)
    stainMatrix = controller._stainMatrixOrdering(dictionary)

    #-- Concentration estimation by Lasso minimization
    currentParameters = {}
    currentParameters.update([(key, spamsParameters[key]) for key in controller.spams_lasso_legal if key in spamsParameters])
    concentrations = spams.lasso(np.asfortranarray(I_od, dtype=controller.DTYPE), D=stainMatrix, **currentParameters) #-- Concentration estimation by Lasso minimization
    concentrations = controller._spams_reshapeSparseEstimation(concentrations) #-- Reshaping of Lasso output of SPAMS to get a regular numpy array

    I_od_noise = I_od - controller.HE_ref.dot(concentrations) #-- Noise computation

    #-- Reference scaling of concentration ranges
    concentrations = controller._concentrationContrastNormalize(concentrations)

    #-- Normalize stain vectors by changing vector base
    I_od_norm = controller.HE_ref.dot(concentrations)

    #-- Conversion back to RGB space
    I_rgb_norm = controller._od2rgb(I_od_norm)
    I_rgb_norm = np.array(I_rgb_norm, dtype=np.uint8) #-- Uint8 type conversion
    I_rgb_norm = controller._1Dto2D(I_rgb_norm) #-- 2D conversion

    #I_rgb_noise = self._od2rgb(I_od_noise)
    #I_rgb_noise =np.array(I_rgb_noise, dtype=np.uint8) #-- Uint8 type conversion
    I_od_noise = controller._1Dto2D(I_od_noise)


    return I_rgb_norm, stainMatrix, concentrations, I_od_noise
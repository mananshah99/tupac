#r -*- coding: utf-8 -*-
"""
Created in May 2016
@author: mlafarge

Python implementation of Macenko's method for staining normalization 2009

References:
    MACENKO'S METHOD:
    A method for normalizing histology slides for quantitative analysis. M.
    Macenko et al., ISBI 2009

    Inspired by Veta's Matlab implementation, 2013
    https://github.com/mitkovetta/staining-normalization
"""

import copy
import numpy as np

DEFAULT_PARAMETERS = {
    "alpha": 0.01, # percentage tolerance for the pseudo-min and pseudo-max angles of fitting plane vectors
    "beta": 0.99 # percentage tolerence for maximum concentration selection
}

def printif(s):
    print(s)

def macenko(controller,
    alpha = DEFAULT_PARAMETERS["alpha"],
    beta = DEFAULT_PARAMETERS["beta"]):#-- percentile parameter
    """
    Macenko's method for staining normalization
    INPUT:
        (StainingController) instance for global normalization parameter state
    OUTPUT:
        (numpy array) I_rgb_norm:     normalized image
        (numpy array) stainMatrix:    unmixing matrix (stain vectors as columns)
        (numpy array) concentrations: concentration maps
        (numpy array) I_od_noise:     map of the noise in OD space

    """
    import time
    t00 = time.time()
    t0 = time.time()
    I_rgb = copy.deepcopy(controller._image) #-- Copy of the original image
    t1 = time.time()
    #print("copy.deepcopy: ", t1-t0) 
    #-- Image reshaping: list of pixel vectors
    I_rgb = controller._2Dto1D(I_rgb)

    #-- Optical Density convertion
    I_od = controller._rgb2od(I_rgb)

    #-- Transparent pixels thresholding
    I_odT = controller._odThresholding(I_od)

    t0=time.time()
    #-- Calculate eigenvectors
    I_od_cov = np.cov(I_odT) #-- Covariance matrix
    I_eigVal, I_eigVect = np.linalg.eigh(I_od_cov) #-- Eigen analysis
    
    I_eigVect = np.multiply(I_eigVect, 1-2*np.all(I_eigVect <=0, axis=0)) #-- Potential sign swapping

    #for k in range(3):
    #    print np.array([1,1,1]).dot(I_eigVect[:,k])
    #raw_input()

    sorted_idx = np.argsort(I_eigVal)[::-1] #-- Reversed sorted list
    t1 = time.time()

    #print("eigenvector calc: ", t1-t0)

    t0=time.time()
    #-- Projection on the plane spanned by the eigenvectors
    #-- corresponding to the two largest eigenvalues
    projection_matrix = np.array([I_eigVect[:,idx] for idx in sorted_idx[0:2]], dtype=controller.DTYPE) #-- Projection matrix
    #print "Proj matrix:", projection_matrix
    I_od_proj = projection_matrix.dot(I_odT) #-- Projection weights
    t1=time.time()
    #print("projection time: ", t1-t0)

    #-- Projected pixel angle computation
    #print "Proj weights :", I_od_proj[0,0:20], " by ", I_od_proj[1,0:20]
    phi = np.arctan2(I_od_proj[0,:], I_od_proj[1,:]) #-- Compute all the angles
    #print "Angles list :", np.pi - phi[0:20]
    sorted_idx = np.argsort(phi)

    #-- alpha-th maximum angles
    idx_min, idx_max = int((1.0-alpha)*len(sorted_idx)), int(alpha*len(sorted_idx))

    phi_min, phi_max = phi[sorted_idx[idx_min]], phi[sorted_idx[idx_max]] #-- alpha-th lowest and alpha-th highest angles
    #print "TEST angles:", np.pi -phi_min, "  ", np.pi -phi_max


    #-- Stain vector estimations
    stainMatrix_proj = np.array([[np.sin(phi_min), np.sin(phi_max)], [np.cos(phi_min), np.cos(phi_max)]], dtype=controller.DTYPE) #-- Matrix of the estimated stain vectors in the fit plane
    stainMatrix = np.transpose(projection_matrix).dot(stainMatrix_proj) #-- Projection back to OD space

    stainMatrix = controller._stainMatrixOrdering(stainMatrix)
    #print "Project mat:", np.transpose(projection_matrix)
    #print "Stain matrix:", stainMatrix

    #-- Concentration maps computed as a projection on stain vectors
    # concentrations = np.transpose(stainMatrix).dot(I_od)
    concentrations = np.linalg.pinv(stainMatrix).dot(I_od)
    concentrations = np.multiply(concentrations>0, concentrations)

    #I_od_noise = I_od - controller.HE_ref.dot(concentrations) #-- Noise computation
    I_od_noise = None

    #-- Reference scaling of concentration ranges
    sorted_idx = np.argsort(concentrations, axis=1) #-- Sorted concentrations
    idx_max = int(beta*sorted_idx.shape[1])
    maxC = [concentrations[k, idx] for k,idx in enumerate(sorted_idx[:,idx_max])] #-- maximum concentration (beta-percentile)

    #-- alpha-th maximum angles
    concentrations = controller._concentrationContrastNormalize(concentrations, maxC)
    #print "Normalized OD:", concentrations[0,0:20]

    #-- Normalize stain vectors by changing vector base
    I_od_norm = controller.HE_ref.dot(concentrations)
    
    #-- Conversion back to RGB space
    I_rgb_norm = controller._od2rgb(I_od_norm)
    #print "Normalized RGB:", I_rgb_norm[0,0:20]

    I_rgb_norm = np.array(I_rgb_norm, dtype=np.uint8) #-- Uint8 type conversion
    I_rgb_norm = controller._1Dto2D(I_rgb_norm) #-- 2D conversion

    #I_rgb_noise = self._od2rgb(I_od_noise)
    #I_rgb_noise = np.array(I_rgb_noise, dtype=np.uint8) #-- Uint8 type conversion
    #I_od_noise = controller._1Dto2D(I_od_noise)
    I_od_noise = None
    #print("total: ", time.time() - t00)
    return I_rgb_norm, stainMatrix, concentrations, I_od_noise

# -*- coding: utf-8 -*-
"""
Created in May 2016
@author: mlafarge

File of default constants used by StainingNormalizer required by all the methods
Contains also default parameters for used libraries.
"""


## Global default parameters
DEFAULT_INTENSITY_MAX = 240                                                  # Transmitted light intensity
DEFAULT_OPTICAL_DENSITY_MIN_THRESHOLD = 0.15                                 # OD threshold for transparent pixels
DEFAULT_HE_REF_0 = [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]    # Reference H&E stain vecotrs matrix (respective column vectors)
DEFAULT_HE_REF_0_MAX_CONCENTRATION = [1.9705, 1.0308]                        # Reference maximum H&E stain concentrations
# Alternative parameters
DEFAULT_HE_REF_1 = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
DEFAULT_HE_REF_1_MAX_CONCENTRATION = [4.3, 6.3]
#DEFAULT_HE_REF_1_MAX_CONCENTRATION = [3.6, 5.5]

#- List of legal parameters for SPAMS library
DEFAULT_SPAMS_DL_LEGAL = ["K", "return_model", "posAlpha", "posD", "lambda1", "batchsize", "iter"]
DEFAULT_SPAMS_LASSO_LEGAL = ["L","lambda1","lambda2","mode","pos","ols","numThreads","length_path","verbose","cholesky"]

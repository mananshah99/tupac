# -*- coding: utf-8 -*-
"""
Created in May 2016
@author: mlafarge

Example to compare different staining normalization algorithms
"""
##
##-- Libraries of interest
##
import sys
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import spams
import copy
import time
import math

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import StainingNormalizer as SN #-- Staining normalizer


def __main():
    #path, x,y = '/utrecht_428044_r1_p0002.tif', 100, 100 #-- Path to the input image
    path, x,y = os.path.join(sys.path[0], "USCB_001.tif"), 200, 200 #-- Path to the input image

    img = cv2.imread(path) #-- Read image
    img[:,:,(0,2)] = img[:,:,(2,0)] #-- RGB representation
    img = img[y:y+256,x:x+256,:]

    T = SN.Controller(img)
    print "... done."

    Macenko_params = SN.method.macenko_p #-- Macenko's normalization
    Macenko_decomposition = T(SN.method.macenko)

    Vahadane_params = SN.method.vahadane_p #-- Vahadane's normalization
    Vahadane_decomposition = T(SN.method.vahadane)
    results = [Macenko_decomposition, Vahadane_decomposition] #-- List of data to plot
    print "//"

    #--
    #-- Visualization
    #--
    F = plt.figure() #-- Main figure initialization
    F.canvas.set_window_title('Example')
    #plt.get_current_fig_manager().window.wm_geometry("600x600+400+20")
    F_list = {}

    _plot = lambda x=0,y=0,rs=1,cs=1, **kwargs: plt.subplot2grid((3,4), (x,y), rowspan=rs, colspan=cs, **kwargs)
    _imshowParams = {"interpolation":'nearest', "aspect":'equal'}

    ##--
    ##-- ORIGINAL IMAGE
    F_list["original"] = _plot(0, 0, 1, 1)
    sharedAxes = {"sharex":F_list["original"], "sharey":F_list["original"]}
    plt.imshow(img, **_imshowParams) #-- [:,:,1] , cmap='Greys_r'
    plt.gca().set_title("Original", fontsize=10)

    ##--
    ##-- GLOBAL INFORMATION
    #-- Text information
    ax = _plot(0, 1, 1, 4, frameon=False) #-- New subplot ## , projection="3d"
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    text = "Normalization methods comparison"
    text += "\nMacenko parameters: {}".format(Macenko_params)
    text += "\nVahadane parameters: {}".format(Vahadane_params)
    text += "\n"

    angle_dev = np.arccos((results[0][1][:,0].T).dot(results[1][1][:,0])/(np.linalg.norm(results[0][1][:,0])*np.linalg.norm(results[1][1][:,0])))
    angle_dev = 180 * angle_dev / np.pi
    angle_dev = np.around(angle_dev, decimals=2)
    dist = np.around(np.abs(results[0][1][:,0]-results[1][1][:,0]), decimals=2).T
    text += "\nSt.Vector H difference: {} - angle dev: {}".format(dist, angle_dev)

    angle_dev = np.arccos((results[0][1][:,1].T).dot(results[1][1][:,1])/(np.linalg.norm(results[0][1][:,1])*np.linalg.norm(results[1][1][:,1])))
    angle_dev = 180 * angle_dev / np.pi
    angle_dev = np.around(angle_dev, decimals=2)
    dist = np.around(np.abs(results[0][1][:,1]-results[1][1][:,1]), decimals=2).T	
    text += "\nSt.Vector E difference: {} - angle dev: {}".format(dist, angle_dev)
    text += "\n\n\n"

    ax.text(0,0,text, fontsize=10)

    ##--
    ##-- PLOT ALL THE RESULTS AT A DIFFERENT ROW FOR EACH METHOD
    for k, result in enumerate(results):
        #-- Reorganization of variables
        method_name = [SN.method.macenko.__name__, SN.method.vahadane.__name__][k]
        I_rgb_norm, stainMatrix, concentrations, I_od_noise = result
        currentRow = k+1

        #--Additionnal information
        MSE = np.sum(np.square(I_od_noise))/(I_od_noise.shape[0]*I_od_noise.shape[1])

        #-- Draw sub-plots
        currentPlotList = {}
        currentPlotList[method_name+"original"] = _plot(currentRow, 0, 1, 1,**sharedAxes)
        plt.imshow(I_rgb_norm, **_imshowParams) #-- [:,:,1] , cmap='Greys_r'
        plt.gca().set_title("{} HE normalization".format(method_name), fontsize=10)

        currentPlotList[method_name+"_H"] = _plot(currentRow, 1, 1, 1,**sharedAxes)
        demo_stain = stainMatrix[:,0].reshape(3,1).dot(concentrations[0,:].reshape(1, concentrations.shape[1])) #-- Concentration projection
        demo_stain = T._od2rgb(demo_stain)
        demo_stain = T._1Dto2D(demo_stain)
        demo_stain = np.array(demo_stain, dtype=np.uint8) #-- Uint8 convertion
        plt.imshow(demo_stain, **_imshowParams) #-- [:,:,1] , cmap='Greys_r'
        plt.gca().set_title("{} H-only".format(method_name)+"\n[{},{},{}]".format(*[round(el,2) for el in stainMatrix[:,0]]), fontsize=9)

        currentPlotList[method_name+"_E"] = _plot(currentRow, 2, 1, 1,**sharedAxes)
        demo_stain = stainMatrix[:,1].reshape(3,1).dot(concentrations[1,:].reshape(1, concentrations.shape[1])) #-- Concentration projection
        demo_stain = T._od2rgb(demo_stain)
        demo_stain = T._1Dto2D(demo_stain)
        demo_stain = np.array(demo_stain, dtype=np.uint8) #-- Uint8 convertion
        plt.imshow(demo_stain, **_imshowParams) #-- [:,:,1] , cmap='Greys_r'
        plt.gca().set_title("{} E-only".format(method_name)+"\n[{},{},{}]".format(*[round(el,2) for el in stainMatrix[:,1]]), fontsize=9)

        currentPlotList[method_name+"_err"] = _plot(currentRow, 3, 1, 1, **sharedAxes)
        plt.imshow(I_od_noise, cmap='Greys_r', **_imshowParams) #-- [:,:,1] , cmap='Greys_r'
        plt.gca().set_title("{} OD map \nreconstruction error: {}".format(method_name, round(MSE,3)), fontsize=10)

        for ax in currentPlotList:
            ax = currentPlotList[ax]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.savefig('
    F.show() #-- Displays figure


__main()
v = input()

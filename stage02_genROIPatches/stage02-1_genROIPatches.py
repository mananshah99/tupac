#!/usr/bin/env python
# coding: utf-8
#
# Copyright © 2016 Dayong Wang <dayong.wangts@gmail.com> and David Zhu
#
# Distributed under terms of the MIT license.

from __future__ import print_function, division

import os,sys,argparse
import logging
import openslide as osi
import skimage as ski
import skimage.io as skio
#import tools
import numpy as np
import time
from multiprocessing import Pool
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import decimal

DESCRIPTION = """
"""

def genInds(dataLoc, ID, inputLevel):
    ROIs = np.genfromtxt(dataLoc + 'ROI/ROI/TUPAC-TR-' + ID + '-ROI.csv', delimiter=",")

    numRegions = len(ROIs[:,0])
    ind_h = []
    ind_w = []

    for i in range(0, numRegions):
        cur_x, cur_y, cur_w, cur_h = to_higher_level(0, inputLevel, ROIs[i,:])

        for j in range(cur_x, cur_x + cur_w):
            for k in range(cur_y, cur_y + cur_h):
                ind_h.append(k)
                ind_w.append(j)
#    print(len(ind_h))

    return np.asarray(ind_h), np.asarray(ind_w)

def genIDList(dataLoc):
    IDList = []
    for name in os.listdir(dataLoc + 'ROI/ROI/'):
#    for name in os.listdir(dataLoc + 'TrainingData/training_image_data/'):
        IDname = name[9:12]
        if IDname == '205':
            continue
        IDList.append(IDname)
    return IDList

def to_higher_level(in_level, out_level, values):
    newvalues = []
    scale = np.power(levelpow, out_level - in_level)
    for value in values:
        newvalues.append(int(value/scale))
    return newvalues

def extend_inds_to_level0(input_level, h, w):
    gap = input_level - 0
    v = np.power(levelpow, gap)
    hlist = h * v + np.arange(v)
    wlist = w * v + np.arange(v)
    hw = []
    for hv in hlist:
        for wv in wlist:
            hw.append([hv, wv])
    return hw

def get_tl_pts_in_level0(OUT_LEVEL, h_level0, w_level0, wsize):
    scale = np.power(levelpow, OUT_LEVEL)
    wsize_level0 = wsize * scale
    wsize_level0_half = wsize_level0 / 2

    h1_level0, w1_level0 = h_level0 - wsize_level0_half, w_level0 - wsize_level0_half
    return int(h1_level0), int(w1_level0)

def get_image(wsi, h1_level0, w1_level0, OUT_LEVEL, wsize):
    img = wsi.read_region(
            (w1_level0, h1_level0),
            OUT_LEVEL, (wsize, wsize))
    img = np.asarray(img)[:,:,:3]
    return img

def getAFolder(folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    return folderName

def getFolders(folder, subFolders):
    for subFolder in subFolders:
        if not os.path.exists('%s/%s'%(folder, subFolder)):
            os.mkdir('%s/%s'%(folder, subFolder))

def genPatchesPar(params):
    ID, srcType, imgType, imgNum, wsize, inputLevel, outputLevel, outputFolder = params
    genPatches(ID, srcType, imgType, imgNum, wsize, inputLevel, outputLevel, outputFolder)

def genPatches(ID, srcType, imgType, imgNum, wsize, inputLevel, outputLevel, outputFolder='training_data'):
    ####
    print("**************************---ID = %s---***********************"%ID)
    wsiName = dataLoc + 'TrainingData/training_image_data/TUPAC-TR-' + ID + '.svs'
    wsi = osi.open_slide(wsiName)
    print(wsi.level_dimensions)

    wsiMaskName = dataLoc + 'TrainingData/small_images-level-2-mask/TUPAC-TR-' + ID + '.png'
    wsi_mask = skio.imread(wsiMaskName, True)
    label = 1

    ind_h, ind_w = genInds(dataLoc, ID, inputLevel)
    if imgType == "TN":
        msk = wsi_mask
        for i in range(0, len(ind_h)):
            curh = ind_h[i]
            curw = ind_w[i]
            msk[curh][curw] = 0
        ind_h, ind_w = np.nonzero(msk>0)

    rinds = np.random.permutation(len(ind_h))
    ind_h = ind_h[rinds]
    ind_w = ind_w[rinds]
    ind_h_size = len(ind_h)

    imgList = []

    getAFolder('%s/img-%s/%s_%s_l%02d'%(outputFolder, srcType, srcType, ID, label))
    if 1:
        getAFolder('%s/msk/%s_%s_l%02d'%(outputFolder, srcType, ID, label))
#################################################################################################################################
    for patch_idx, (ch, cw) in enumerate(zip(ind_h, ind_w)):
        # all the pixels in chcw_list are marked with 1
        chcw_level0_list = extend_inds_to_level0(inputLevel, ch, cw) # get the "square" of pixels matched from inputLevel to 0
        idx = int(len(chcw_level0_list) / 2)
        chcw_level0 = chcw_level0_list[idx] # take middle point

        h_level0, w_level0 = chcw_level0
        h1_level0, w1_level0 = get_tl_pts_in_level0(outputLevel, h_level0, w_level0, wsize) # should give top left corner of patch

        img_name = '%s/img-%s/%s_%s_l%02d/%s_%s_l%02d_h%010d_w%010d_i%05d.png'%(outputFolder, srcType, srcType, ID, label, srcType, ID, label, ch, cw, idx)
        img = get_image(wsi, h1_level0, w1_level0, outputLevel, wsize)
        percentage = 0.0
        if 1:
            h1_level2, w1_level2, wsize2 = to_higher_level(0, 2, (h1_level0, w1_level0, wsize))
#            print("h1_level0, w1_level0, wsize: %d, %d, %d"%(h1_level0, w1_level0, wsize))
#            print("h1_level2, w1_level2, wsize2: %d, %d, %d"%(h1_level2, w1_level2, wsize2))
#            print(wsi_mask)
#            print(wsi_mask[h1_level2, w1_level2])
            img_mask = wsi_mask[h1_level2:h1_level2 + wsize2, w1_level2:w1_level2 + wsize2]
#            img_mask = get_image(wsi_mask, h1_level0, w1_level0, outputLevel, wsize)
#            print(img_mask)
            img_mask[img_mask>0]=1
#            print(img_mask)
#            print("sum is %d"%np.sum(img_mask))
            percentage = np.sum(img_mask) / float((wsize2 * wsize2)) # what percent of the image is tissue
#            img_mask = img_as_ubyte(rgb2gray(img_mask))
#            img_mask[img_mask>0]=1
#            img_mask_name = '%s/msk/%s_%03d_l%02d/%s_%03d_l%02d_h%010d_w%010d_i%05d.png'%(outputFolder, srcType, ID, label, srcType, ID, label, ch, cw, idx)

        if imgType == "TP":
            if percentage > 0.50: # want > 50% tissue
                strline = '%s %d %.2f' % (img_name, 1, percentage)
#                print('saving to ' + img_name)
                skio.imsave(img_name, img)
#                skio.imsave(img_mask_name, img_mask)
                imgList.append(strline)
                strinfo='ok'
            else:
                strinfo='ignore'
        elif imgType == "TN":
            if percentage >= 0:
                strline = '%s %d %.2f' % (img_name, 0, percentage)
                skio.imsave(img_name, img)
                imgList.append(strline)
                strinfo='ok'
            else:
                strinfo='ignore'
        elif imgType == "NN":
            strline = '%s %d %.2f' % (img_name, 0, 0)
            skio.imsave(img_name, img)
            imgList.append(strline)
            strinfo='ok'

        print("%d (%d)-> %s %d: (y:%d,x:%d) wsize:%d perc:%.2f, %s"%(patch_idx, ind_h_size, wsiName, label, ch, cw, wsize, percentage, strinfo))
        # we got enough patches
        if len(imgList) > imgNum:
            break
    ####
    listName = '%s/lst/%s_%s_l%02d.lst'%(outputFolder, srcType, ID, label)
    with open(listName, 'w') as f:
        f.write('\n'.join(imgList))

    time.sleep(5)
    wsi.close()
    if 0:
        wsi_mask.close()
    return imgList

threadNUM = 50
imgNum = 999
#imgNum=10
#wsize=256
#wsize=128
wsize=256
#wsize=32
inputLevel=2
outputLevel=0
levelpow = 4

dataLoc = '/data/dywang/Database/Proliferation/data/'

outputFolder = getAFolder('/data/dzhu1/LEVEL%02d/sample_W%03d_P%010d'%(outputLevel, wsize, imgNum))
getFolders(outputFolder, ['img-ROI', 'img-Normal', 'msk', 'lst'])

IDList = genIDList(dataLoc)

if 0:
    IDList = []
    for i in range(1, 501):
        IDList.append('%03d'%i)

    #print(IDList)
    dimlist = []
    for ID in IDList:
        wsiName = dataLoc + 'TrainingData/training_image_data/TUPAC-TR-' + ID + '.svs'
        wsi = osi.open_slide(wsiName)
        dims = wsi.level_dimensions
        curlist = [ID]
        for i in range(0, wsi.level_count-1):
            dec = decimal.Decimal(dims[i][0] / dims[i+1][0])
            curlist.append(int(round(dec, 0)))
        dimlist.append(curlist)

if 0:
    srcType = 'ROI'
    imgType = 'TP'
    params = []
    for ID in IDList:
        params_i = [ID, srcType, imgType, imgNum, wsize, inputLevel, outputLevel, outputFolder]
        params.append(params_i)
    if 1:
        pool = Pool(threadNUM)
        pool.map(genPatchesPar, params)
    else:
        genPatchesPar(params[0])

if 1:
    srcType = 'Normal'
    imgType = 'TN'
    params = []
    for ID in IDList:
        params.append([ID, srcType, imgType, imgNum, wsize, inputLevel, outputLevel, outputFolder])
    if 1:
        pool = Pool(threadNUM)
        pool.map(genPatchesPar, params)
    else:
        genPatchesPar(params[0])

if 0: # not used
    srcType = 'Normal'
    imgType = 'NN'
    params = []
    for ID in range(1, 160 + 1):
        params.append([ID, srcType, imgType, imgNum, wsize, inputLevel, outputLevel, outputFolder])
    if 1:
        pool = Pool(threadNUM)
        pool.map(genPatchesPar, params)
    else:
        genPatchesPar(params[0])

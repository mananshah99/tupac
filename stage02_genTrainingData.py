#!/usr/bin/env python
# coding: utf-8
#
# Copyright © 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function, division

import os,sys,argparse
import logging
import openslide as osi
import skimage as ski
import skimage.io as skio
import tools
import numpy as np
import time
from multiprocessing import Pool
from skimage.color import rgb2gray
from skimage import img_as_ubyte


DESCRIPTION = """
"""

def extend_inds_to_level0(input_level, h, w):
    gap = input_level - 0
    v = np.power(2, gap)
    hlist = h * v + np.arange(v)
    wlist = w * v + np.arange(v)
    hw = []
    for hv in hlist:
        for wv in wlist:
            hw.append([hv, wv])
    return hw

def get_tl_pts_in_level0(OUT_LEVEL, h_level0, w_level0, wsize):
    scale = np.power(2, OUT_LEVEL)
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
        os.mkdir(folderName)
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
    wsiName = '../Train_%s/%s_%03d.tif'%(srcType, srcType, ID)
    wsi = osi.open_slide(wsiName)
    label = 0
    if imgType == "TP": # Tumor P
        wsiMaskName = '../Mask/%s_%03d_Mask.tif'%(srcType, ID)
        wsi_mask = osi.open_slide(wsiMaskName)
        mskName = '../LEVEL_%03d/%s_Mask_Truth/%s_%03d.png'%(inputLevel, srcType, srcType, ID)
        msk = skio.imread(mskName, True) # the ground truth mask
        label = 1
    elif imgType == "TN": # Tumor N
        wsiMaskName = '../Mask/%s_%03d_Mask.tif'%(srcType, ID)
        wsi_mask = osi.open_slide(wsiMaskName)
        mskName = '../LEVEL_%03d/%s_Mask/%s_%03d.png'%(inputLevel, srcType, srcType, ID)
        msk1 = skio.imread(mskName, True) > 0 # the tissue mask
        mskName = '../LEVEL_%03d/%s_Mask_Truth_R200/%s_%03d.png'%(inputLevel, srcType, srcType, ID)
        msk2 = skio.imread(mskName, True) > 0 # the surronding region of tumor
        msk = np.logical_and(msk1, msk2) # not a tumor but a tissue
        #skio.imsave('b.png', msk.astype(np.float32))
    elif imgType == "NN": # normal N
        wsi_mask = None
        mskName = '../LEVEL_%03d/%s_Mask/%s_%03d.png'%(inputLevel, srcType, srcType, ID)
        msk = skio.imread(mskName, True)

    ind_h, ind_w = np.nonzero(msk>0)
    rinds = np.random.permutation(len(ind_h))
    #ind_h = ind_h[rinds[:imgNum]]
    #ind_w = ind_w[rinds[:imgNum]]
    ind_h = ind_h[rinds]
    ind_w = ind_w[rinds]
    ind_h_size = len(ind_h)

    imgList = []

    getAFolder('%s/img/%s_%03d_l%02d'%(outputFolder, srcType, ID, label))
    if wsi_mask:
        getAFolder('%s/msk/%s_%03d_l%02d'%(outputFolder, srcType, ID, label))

    for patch_idx, (ch, cw) in enumerate(zip(ind_h, ind_w)):
        # all the pixels in chcw_list are marked with 1
        chcw_level0_list = extend_inds_to_level0(inputLevel, ch, cw)
        idx = int(len(chcw_level0_list) / 2)
        chcw_level0 = chcw_level0_list[idx]

        h_level0, w_level0 = chcw_level0
        h1_level0, w1_level0 = get_tl_pts_in_level0(outputLevel, h_level0, w_level0, wsize)

        img_name = '%s/img/%s_%03d_l%02d/%s_%03d_l%02d_h%010d_w%010d_i%05d.png'%(outputFolder, srcType, ID, label, srcType, ID, label, ch, cw, idx)
        img = get_image(wsi, h1_level0, w1_level0, outputLevel, wsize)
        percentage = 0.0
        if wsi_mask:
            img_mask = get_image(wsi_mask, h1_level0, w1_level0, outputLevel, wsize)
            percentage = np.sum(img_mask[:,:,0] / 255) / float((wsize * wsize))
            img_mask = img_as_ubyte(rgb2gray(img_mask))
            img_mask[img_mask>0]=1
            img_mask_name = '%s/msk/%s_%03d_l%02d/%s_%03d_l%02d_h%010d_w%010d_i%05d.png'%(outputFolder, srcType, ID, label, srcType, ID, label, ch, cw, idx)

        if imgType == "TP": # the overlapping should be larger than 0.95
            if percentage > 0.95:
                strline = '%s %d %.2f' % (img_name, 1, percentage)
                skio.imsave(img_name, img)
                skio.imsave(img_mask_name, img_mask)
                imgList.append(strline)
                strinfo='ok'
            else:
                strinfo='ignore'
        elif imgType == "TN":
            if percentage >= 0 and percentage < 0.95:
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
    listName = '%s/lst/%s_%03d_l%02d.lst'%(outputFolder, srcType, ID, label)
    with open(listName, 'w') as f:
        f.write('\n'.join(imgList))

    time.sleep(5)
    wsi.close()
    if wsi_mask:
        wsi_mask.close()
    return imgList

threadNUM = 20
imgNum=10 * 1000
#imgNum=10
#wsize=256
#wsize=128
wsize=64
#wsize=32
inputLevel=5
outputLevel=0

outputFolder = getAFolder('../training_data/LEVEL%02d/sample_W%03d_P%010d'%(outputLevel, wsize, imgNum))
getFolders(outputFolder, ['img', 'msk', 'lst'])
if 1:
    srcType = 'Tumor'
    imgType = 'TP'
    params = []
    for ID in range(1, 110 + 1):
        params_i = [ID, srcType, imgType, imgNum, wsize, inputLevel, outputLevel, outputFolder]
        params.append(params_i)
    if 1:
        pool = Pool(threadNUM)
        pool.map(genPatchesPar, params)
    else:
        genPatchesPar(params[0])

if 1:
    srcType = 'Tumor'
    imgType = 'TN'
    params = []
    for ID in range(1, 110 + 1):
        params.append([ID, srcType, imgType, imgNum, wsize, inputLevel, outputLevel, outputFolder])
    if 1:
        pool = Pool(threadNUM)
        pool.map(genPatchesPar, params)
    else:
        genPatchesPar(params[0])

if 1:
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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cPickle as pickle
import os
import glob
import sys

sys.setrecursionlimit(80000)

import numpy as np

from sklearn.utils import shuffle
from scipy.spatial import distance
from sklearn import cross_validation
import random
import csv

import skimage as ski
import skimage.io as skio
import skimage.util as skiu
import random

def getAFolder(folderName):
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    return folderName

def getFolders(folder, subFolders):
    for subFolder in subFolders:
        if not os.path.exists('%s/%s'%(folder, subFolder)):
            os.mkdir('%s/%s'%(folder, subFolder))

def image_pad(img, pad_size, modev):
    if len(img.shape) == 2:
        return skiu.pad(img, pad_width=pad_size, mode=modev)
    if len(img.shape) > 2:
        z = len(img.shape)
        imgs = []
        for i in range(z):
            img_i = np.squeeze(img[:,:, i])
            img_i = skiu.pad(img_i, pad_width=pad_size, mode=modev)
            imgs.append(img_i)
        nimg = np.stack(imgs, axis=2)
        return nimg

def get_motesis_set(imgfile, inputFolder, stage1Folder):
    def update_mask(msk, ref):
        w1, h1 = ref.shape
        w2, h2 = msk.shape
        if w2 > w1 and h2 > h1:
            w3 = int((w2-w1)/2.0)
            h3 = int((h2-h1)/2.0)
            return msk[h3:h3+h1, w3:w3+w1]
        else:
            return msk

    imgfileRoot = imgfile.split('.')[0]
    annotfile = csv.reader(open('%s/%s.csv'%(inputFolder, imgfileRoot), 'rb'))
    msk1 = skio.imread('%s/%s_mask.png'%(inputFolder, imgfileRoot), True) > 0
    msk2 = skio.imread('%s/%s_mask_nuclei.png'%(inputFolder, imgfileRoot), True) > 0 # candidate samples
    msk3 = None
    if stage1Folder:
        msk3 = skio.imread('%s/%s.png'%(stage1Folder, imgfileRoot), True) > 0 # prediction results from stage 1
        msk3 = update_mask(msk3, msk2)

    centroids, pos_coords_sel = [], []
    for row_id, row in enumerate(annotfile):
        coords = []
        for i in range(0, len(row)/2):
            xv, yv = (int(row[2*i]), int(row[2*i+1]))
            coords.append([yv, xv])
        centroid = np.array(coords).mean(axis=0).astype(int)
        centroids.append(centroid)
        #print row
        print row_id, centroid
        for i in range(0, len(row)/2):
            xv, yv = (int(row[2*i]), int(row[2*i+1]))
            if distance.euclidean([yv, xv], centroid) <= RADIUS:
                pos_coords_sel.append((yv, xv))
    print "#selected postive points:", len(pos_coords_sel)

    msk_nm_neg = msk2 > msk1
    Ys, Xs  = np.where(msk_nm_neg)
    nm_neg_coords = [[y, x] for y, x in zip(Ys, Xs)]
    if len(nm_neg_coords) > 2*len(pos_coords_sel):
        nm_neg_coords = random.sample(nm_neg_coords, 2*len(pos_coords_sel))
    print "#nm_neg", len(nm_neg_coords)

    fp_neg_coords = None
    if msk3:
        msk_fp_neg = msk3 > msk1
        Ys, Xs = np.where(msk_fp_neg)
        fp_neg_coords = [[y, x] for y, x in zip(Ys, Xs)]
        if len(fp_neg_coords) > len(pos_coords_sel):
            fp_neg_coords = random.sample(fp_neg_coords, len(pos_coords_sel))
        print "#fp_neg", len(fp_neg_coords)

    return pos_coords_sel, nm_neg_coords, fp_neg_coords

def get_patches(imgfile, coords, typev, inputFolder, outputFolder, patchsize):
    if not coords:
        return []
    padsize = int(patchsize/2.0)
    imgfileRoot = imgfile.split('.')[0]

    img = skio.imread('%s/%s.bmp'%(inputFolder, imgfileRoot))
    msk = skio.imread('%s/%s_mask.png'%(inputFolder, imgfileRoot), True)
    img = image_pad(img, padsize, modev='reflect')
    msk = image_pad(msk, padsize, modev='constant')

    outputFolderSub = getAFolder('%s/%s'%(outputFolder, imgfileRoot))

    lists = []
    for cid, coord in enumerate(coords):
        y_ori, x_ori = coord
        y = y_ori + padsize
        x = x_ori + padsize

        img_sub = img[(y - patchsize/2):(y + patchsize/2), (x - patchsize/2):(x + patchsize/2),:]
        msk_sub = msk[(y - patchsize/2):(y + patchsize/2), (x - patchsize/2):(x + patchsize/2)]

        getFolders(outputFolderSub, ['patches', 'masks', 'masks_visual'])
        patch_name_str = '%s_TP%02d_ID%010d_Y%05d_X%05d'%(imgfileRoot, typev, cid, y_ori, x_ori)
        img_name = '%s/patches/%s.png'%(outputFolderSub, patch_name_str)
        msk_name1 = '%s/masks/%s.png'%(outputFolderSub, patch_name_str)
        msk_name2 = '%s/masks_visual/%s.png'%(outputFolderSub, patch_name_str)

        skio.imsave(img_name, img_sub)
        skio.imsave(msk_name1, msk_sub>0)
        skio.imsave(msk_name2, (msk_sub>0)*255)

        lists.append([img_name, msk_name1, typev])
        print typev, img_name
        #break
    return lists


def get_motesis_center_pixel(imagelist, inputFoder, outputFolder, stage1Folder, patchsize):
    list1, list2, list3 = [], [], []
    for imgfile in [l.strip().split()[0] for l in open(imagelist).readlines()]:
        # typev
        # 00 negative
        # 01 false positive
        # 20 postive
        pos_coords, neg_coords_nm, neg_coords_fp = get_motesis_set(imgfile, inputFolder, stage1Folder)
        list1_sub = get_patches(imgfile, pos_coords, 20, inputFolder, outputFolder, patchsize)
        list2_sub = get_patches(imgfile, neg_coords_nm, 0, inputFolder, outputFolder, patchsize)
        list3_sub = get_patches(imgfile, neg_coords_fp, 1, inputFolder, outputFolder, patchsize)
        list1 += list1_sub
        list2 += list2_sub
        list3 += list3_sub
        #break

    with open('%s/imagelist_wlabel.txt'%outputFolder, 'w') as f:
        lines = []
        lines+= ['%s 1'%(itm[0]) for itm in list1]
        lines+= ['%s 0'%(itm[0]) for itm in list2]
        lines+= ['%s 0'%(itm[0]) for itm in list3]
        f.write('\n'.join(lines))
    with open('%s/imagelist_wmask.txt'%outputFolder, 'w') as f:
        lines = []
        lines+= ['%s %s'%(itm[0], itm[1]) for itm in list1]
        lines+= ['%s %s'%(itm[0], itm[1]) for itm in list2]
        lines+= ['%s %s'%(itm[0], itm[1]) for itm in list3]
        f.write('\n'.join(lines))

# Constants
SIZE = 2084
PATCH_SIZE = 50
#PATCH_SIZE = 100
#PATCH_SIZE = 256
PATCH_GAP = 50
RADIUS = 10

inputFolder = 'train'
outputFolder = 'train_patches_050/stage01'
#outputFolder = 'train_patches_100/stage01'
#stage1Folder = 'results/googlenet.train'
stage1Folder = None
get_motesis_center_pixel('train.list', inputFolder, outputFolder, stage1Folder, PATCH_SIZE)

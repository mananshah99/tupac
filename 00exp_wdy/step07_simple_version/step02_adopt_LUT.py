#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from skimage.io import imread, imsave
import random
from tqdm import tqdm
from multiprocessing import Pool

random.seed(1122)

IMG_FD = '/data/dywang/Database/Proliferation/libs/00exp_wdy/step07_simple_version'
LUT_FD = '/home/dywang/Proliferation/data/ColorNorm'
OUT_FD = 'color_norm'
DATA_AUG_N = 5

def apply_lut(patch, lut):
    """ Apply look-up-table to patch to normalize H&E staining. """

    ps = patch.shape  # patch size is (rows, cols, channels)
    reshaped_patch = patch.reshape((ps[0] * ps[1], 3))
    normalized_patch = np.zeros((ps[0] * ps[1], 3), dtype=np.uint8)
    idxs = range(ps[0] * ps[1])
    Index = 256 * 256 * reshaped_patch[idxs, 0] + 256 \
        * reshaped_patch[idxs, 1] + reshaped_patch[idxs, 2]
    normalized_patch[idxs] = lut[Index.astype(int)]
    return normalized_patch.reshape(ps[0], ps[1], 3)

def load_LUTs(lut_path):
    print '\tLoading:', lut_path
    img = imread(lut_path)
    lut = np.squeeze(img)
    print '\tDone'
    return lut

jobs = {}

def data_aug(img, N=10, rot=True, noise=20, contrast=0.025):
    imgs = []
    imgs.append(img)
    for i in range(N-1):
        nimg = np.copy(img).astype(np.float)

        k = 0
        ####
        if rot:
            k = random.randint(0,3)
            nimg = np.rot90(nimg, k)

        ####
        alst, blst = [], []
        for i in range(3):
            if random.random() > 0.5:
                a = 1 + contrast * random.random()
            else:
                a = 1 - contrast * random.random()

            if random.random() > 0.5:
                b = random.randint(0, noise)
            else:
                b =-random.randint(0, noise)
            nimg[:,:,i] = a * nimg[:,:,i] + b
            alst.append(a)
            blst.append(b)
        nimg[nimg<0]=0
        nimg[nimg>255]=255
        imgs.append(nimg.astype(np.uint8))
        #print '@@', k, alst, blst
    return imgs

def gen_new_data(param):
    (lut_path, imgs) = param
    print lut_path, len(imgs)

    lut = load_LUTs(lut_path)

    for img in tqdm(imgs):
        (
            patch_name,
            image_name,
            patch_part_path,
            patch_path,
            lut_path
        ) = get_info(img)

        img = imread(patch_path)
        nimg = apply_lut(img, lut)

        out_fd = '%s/%s' % (OUT_FD, patch_part_path)
        print out_fd
        if not os.path.exists(out_fd):
            os.makedirs(out_fd)

        nimgs = data_aug(nimg, N=DATA_AUG_N)

        for nimg_id, nimg in enumerate(nimgs):
            out_path = '%s/%s_%010d.jpg' % (out_fd, patch_name, nimg_id)
            imsave(out_path, nimg)
        #break

def get_info(img):
    itms = img.strip().split('/')
    patch_name = itms[-1]
    image_name = itms[-2]
    lut_name = image_name.replace('.svs', '_LUT.tif')
    patch_path = '%s/%s' % (IMG_FD, img)
    if img[0] == 't':
        lut_path = '%s/testing_image_data/%s' % (LUT_FD, lut_name)
    else:
        lut_path = '%s/training_image_data/%s' % (LUT_FD, lut_name)
    patch_part_path = img.strip()
    return (
            patch_name,
            image_name,
            patch_part_path,
            patch_path,
            lut_path
        )

if __name__ == '__main__':
    files = [
        'imgs_level01.lst',
        'te_imgs_level01.lst',
    ]

    for f in files:
        imgs = [l.strip() for l in open(f)]

        print "#line=", len(imgs)
        for img in imgs:
            (
                patch_name,
                image_name,
                patch_part_path,
                patch_path,
                lut_path
            ) = get_info(img)

            if os.path.exists(lut_path):
                if not jobs.has_key(lut_path):
                    jobs[lut_path] = []
                jobs[lut_path].append(img)
            #else:
            #    print "missing:", lut_path

        jobs_key = jobs.keys()
        print '#Key=', len(jobs_key)

        if 1:
            params = []
            for key in jobs_key:
                params.append([key, jobs[key]])

            if 0:
                for param in params:
                    gen_new_data(param)
                    break
            else:
                pool = Pool(10)
                pool.map(gen_new_data, params)
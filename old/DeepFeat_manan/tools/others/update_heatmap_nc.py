#!/usr/bin/env python
# coding: utf-8
#
# Copyright © 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function, division
DESCRIPTION = """
"""
import os
import sys
import argparse
import logging
from scipy.io import savemat
from deepfeat import get_extractor
import deepfeat.util as dutil
import openslide as osi
import skimage.io as skio
import numpy as np
from random import sample
from skimage.transform import resize
from scipy import sparse, io
import scipy.ndimage as nd
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.decomposition import PCA
import cPickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.preprocessing import normalize

#
import numpy as np
from scipy import ndimage as nd
import skimage as ski
import skimage.io as skio
from skimage.exposure import rescale_intensity
from skimage import morphology
import scipy.ndimage.morphology as smorphology
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.transform import resize

def normr(Mat):
    """Normalize the rows of the matrix so the L2 norm of each row is 1.
    >>> A = rand(4, 4)
    >>> B = normr(A)
    >>> np.allclose(norm(B[0, :]), 1)
    True
    """
    B = normalize(Mat, norm='l2', axis=1)
    return B

#
# split list into groups
#
def list_split(l, n):
    return [ l[i:i+n] for i in range(0, len(l), n) ]

def gen_heatmap_wsi(extractor, feat_name, classifier, wsiName, maskName, model_level, mask_level, heatmap_level, window_size, augmentation, batch_size, group_size, step_size):
    #
    # data augmentation
    #
    def image_augmentation(img_000):
        if augmentation == 1:
            return [img_000]
        elif augmentation == 6:
            img_090 = nd.rotate(img_000, 90)
            img_180 = nd.rotate(img_090, 180)
            img_270 = nd.rotate(img_180, 270)
            img_fph = np.fliplr(img_000)
            img_fpv = np.flipud(img_000)
            return [img_000, img_090, img_180, img_270, img_fph, img_fpv]
        else:
            return None
    #
    # extract image patches from WSI on LEVEL_0 using the
    #
    def get_image(wsi, h1_ml, w1_ml, window_size_w, window_size_h):
        h1_ml_2_level0 = int(h1_ml * np.power(2, model_level))
        w1_ml_2_level0 = int(w1_ml * np.power(2, model_level))
        img = wsi.read_region(
                (w1_ml_2_level0, h1_ml_2_level0),
                model_level,
                (window_size_w, window_size_h)
                )
        img = np.asarray(img)[:,:,:3]
        imgs = image_augmentation(img)
        return imgs
    #
    # check the overlapping with tissue segmentation, which is helpful to reduce
    # the number of pataches
    #
    def has_overlapping_with_tissue_using_mask(msk, h1_ml, w1_ml, gap_between_mask_model):
        if msk is None:
            return True
        else:
            h1_kl = int(h1_ml / np.power(2, gap_between_mask_model))
            w1_kl = int(w1_ml / np.power(2, gap_between_mask_model))
            window_size_kl = int(window_size / np.power(2, gap_between_mask_model))
            return np.sum(msk[h1_kl:h1_kl + window_size_kl, w1_kl:w1_kl + window_size_kl]) > 0
    #
    # Get prediction results using DL
    # len(v_img) = augmentation * num_patches
    #
    def doExtraction(v_img, batch_size):
        values_list = []
        v_img_list = list_split(v_img, batch_size)
        values = []
        for i, v_img_i in enumerate(v_img_list):
            logging.info("\t\t\t sub-group %d (%d)"%(i, len(v_img_i)))
            #
            values_i = extractor.batch_extract_numpy(v_img_i, [i.strip() for i in feat_name.split(',')])
            #print(values_i[0].shape, values_i[1].shape)
            values_i1 = values_i[0] # get the first feature group
            values_i1 = values_i1[:, 1].reshape(-1) # get the possibility of positive
            values_i1[values_i1<1e-4] = 0.0

            #values_i2 = values_i[2] # get the first feature group
            feat = np.squeeze(values_i[1])
            feat = normr(feat)

            feat_pca = classifier[0].transform(feat)
            values_i2 = classifier[1].predict(feat_pca)
            values_i3 = classifier[1].predict_proba(feat_pca)

            values_i1[values_i2 == 0] = 0
            #for ii in range(values_i1.shape[0]):
            #    print(values_i1[ii], values_i2[ii,:], values_i3[ii])

            values = np.concatenate((values, values_i1))
        values = values.reshape(-1, augmentation) # numpy is row first
        values_m = np.mean(values, axis=1)
        values_m = values_m.reshape(-1)
        return values_m

    def gen_heatmap_batch(wsi, msk, outputName, wsi2):
        # ml -> model level 0, 1, 2, 3, 4, 5
        # kl -> mask level 5
        # hl -> heatmap level 2
        img_w_ml, img_h_ml = wsi.level_dimensions[model_level]
        img_w_hl, img_h_hl = wsi.level_dimensions[heatmap_level]
        heatmap_ori = skio.imread(maskName, True)
        heatmap_ori = heatmap_ori / 255.0# load the existing mask
        #hc_hl_lst, wc_hl_lst= np.nonzero(heatmap_ori > 0.8)
        #inds_list=[]
        roi = heatmap_ori > 0.5
        #roi = morphology.remove_small_objects(roi, 500);
        #skio.imsave('roi.png', roi * 255)
        label_roi = label(roi)
        R = regionprops(label_roi)
        r_h2m = np.power(2, heatmap_level - model_level)
        for idx, R_i in enumerate(R):
            if R_i.area > 300:
                print(idx, R_i.area)
                min_row, min_col, max_row, max_col = R_i.bbox
                h, w = max_row-min_row+1, max_col-min_col+1
                min_row_ml, min_col_ml = min_row * r_h2m, min_col * r_h2m
                h_ml, w_ml = h * r_h2m, w * r_h2m

                sel_msk = msk[min_row:max_row+1,min_col:max_col+1]
                sel_msk = resize(sel_msk, [h_ml, w_ml])
                skio.imsave('%03d_%010d_mask.png'%(idx, int(R_i.area)), sel_msk)

                sel_img = get_image(wsi, min_row_ml, min_col_ml, w_ml, h_ml)
                skio.imsave('%03d_%010d.png'%(idx, int(R_i.area)), sel_img)
                if wsi2:
                    sel_img = get_image(wsi2, min_row_ml, min_col_ml, w_ml, h_ml)
                    skio.imsave('%03d_%010d_mask_ground.png'%(idx, int(R_i.area)), sel_img)
            else:
                label_roi[label_roi==R_i.label] = 0

        skio.imsave('label_roi.png', (label_roi>0) * 255)
        # get patches
        if 0:
            for hc_hl, wc_hl in zip(hc_hl_lst, wc_hl_lst):
                hc_ml, wc_ml = hc_hl * np.power(2, heatmap_level - model_level), wc_hl * np.power(2, heatmap_level - model_level)
                h1_ml, w1_ml = hc_ml - window_size / 2, wc_ml - window_size / 2
                inds_list.append([h1_ml, w1_ml])
            logging.info("\t There are %d patches in total!"%(len(inds_list)))

        # predict batches
        if 0:
            values = np.zeros((len(inds_list),), np.float32)
            inds_list_group = list_split(inds_list, group_size)
            for i, inds_list_group_i in enumerate(inds_list_group):
                v_img = []
                for h1_ml, w1_ml in inds_list_group_i:
                    imgs = get_image(wsi, h1_ml, w1_ml)
                    v_img += imgs
                logging.info("\t\t Processing: group %d:%d (%d) : %s"%(i, len(v_img), len(inds_list_group), wsiName))
                values_i = doExtraction(v_img, batch_size)
                #values_list.append(values_i)
                values[i*group_size:i*group_size + len(inds_list_group_i)] = values_i
                #break
            #values = np.hstack(values_list)
            logging.info("\t Done!")

        # genearte the heatmap
        if 0:
            logging.info("\t Updating heatmap...")
            heatmap = np.dstack((heatmap_ori, np.ones(heatmap_ori.shape)))
            for (h1_ml, w1_ml), v in zip(inds_list, values):
                hc_ml, wc_ml = h1_ml + window_size / 2, w1_ml + window_size / 2
                hc_hl = int(hc_ml / np.power(2, heatmap_level - model_level))
                wc_hl = int(wc_ml / np.power(2, heatmap_level - model_level))

                heatmap[hc_hl, wc_hl, 0] = v
                #window_size_hl = int(window_size / np.power(2, heatmap_level - model_level))
                #h1_hl, w1_hl = int(hc_hl - window_size_hl / 2), int(wc_hl - window_size_hl / 2)
                #ori_patch = heatmap[h1_hl:h1_hl+window_size_hl, w1_hl:w1_hl+window_size_hl, :]
                #l1, c1 = np.split(ori_patch, 2, axis=2)
                #nmask = np.tile(np.array([v, 1]).reshape((1,1,2)), (window_size_hl, window_size_hl, 1))
                #l2, c2 = np.split(nmask, 2, axis=2)
                ### merging
                #c3 = c1 + c2
                #l3 = ((l1 * c1) + l2) / c3
                #heatmap[h1_hl:h1_hl+window_size_hl, w1_hl:w1_hl+window_size_hl, :] = np.dstack((l3, c3))
            logging.info("\t Done!")
        # Save the results
        if 0:
            ## save the heat map ##
            #img_h_kl = img_h_hl / np.power(2, mask_level - heatmap_level)
            #img_w_kl = img_w_hl / np.power(2, mask_level - heatmap_level)
            #heatmap_kl = (resize(heatmap, (img_h_kl, img_w_kl)) * 255).astype(np.uint)
            heatmap = np.squeeze(heatmap[:,:,0])
            heatmap = (heatmap * 255).astype(np.uint)
            skio.imsave(outputName, heatmap)
            #skio.imsave(outputName.repalce('.png', np.concatenate((heatmap_ori, heatmap), axis=1))

    ## begin of function : gen_heatmap_wsi ##
    logging.info("Processing: %s ..." % (wsiName))
    wsiNameRoot = wsiName.split('/')[-1]
    wsi = osi.open_slide(wsiName)
    if wsiNameRoot[0] == 'T':
        wsi2 = osi.open_slide('/home/dywang/CLC/Ground_Truth/Mask/%s' % wsiNameRoot.replace('.tif', '_Mask.tif'))
    else:
        wsi2 = None
    if maskName:
        msk = skio.imread(maskName, True)
    else:
        msk = None
    outputName = '%s/%s'%(args.output_folder, wsiName.split('/')[-1].split('.')[0] + '.png')
    if not os.path.exists(outputName):
        gen_heatmap_batch(wsi, msk, outputName, wsi2)
    else:
        logging.info("Omit : %s"%(outputName))

def create_parser():
    """ Parse program arguments.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=
                                     argparse.RawTextHelpFormatter)
    parser.add_argument('extractor', type=str, help='name of extractor')
    parser.add_argument('conf_file', type=str, help='config file')
    parser.add_argument('feat_name', type=str, help='feature name')
    parser.add_argument('classifier_path', type=str, help='classifier path')
    parser.add_argument('input_folder', type=str, help='output folder')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('model_level', type=int, help='deep model level')
    parser.add_argument('heatmap_level', type=int, default=2, help='the level of heatmap')
    parser.add_argument('mask_level', type=int, default=5, help='mask image level')
    parser.add_argument('--augmentation', type=int, default=1, help='do rotation')
    parser.add_argument('--window_size', type=int, default=256, help='windows size')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--group_size', type=int, default=100, help='group size')
    parser.add_argument('--step_size', type=int, default=100, help='step size')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0],
                        help='config file')
    parser.add_argument('--gpu', action='store_true',
                        help='config file')
    parser.add_argument("--log", type=str, default="INFO", help="log level")
    return parser

def load_classificer(args):
    """ Main entry.
    """
    mats = [l.strip() for l in os.popen('ls %s/mat/*.mat'%args.classifier_path)]
    nams = [l.strip().split()[0] for l in open('%s/lst'%args.classifier_path)]
    lals = [int(l.strip().split()[1]) for l in open('%s/lst'%args.classifier_path)]

    feat_list = []
    for mat in tqdm(mats):
        feat_i = np.squeeze(loadmat(mat)['feat'])
        feat_list.append(feat_i)
    feat = np.vstack(feat_list)
    print("# %d features loaded !" % (feat.shape[0]))
    feat = normr(feat)

    pca_model_path = '%s/pca.pkl'%args.classifier_path
    if not os.path.exists(pca_model_path):
        print("Train PCA...")
        pca = PCA(n_components = 256)
        pca.fit(feat)
        with open(pca_model_path, 'wb') as fid:
            cPickle.dump(pca, fid)
    else:
        print("Load PCA...")
        with open(pca_model_path, 'rb') as fid:
            pca = cPickle.load(fid)
    feat_pca = pca.transform(feat)

    print('# new feature shape:', feat_pca.shape)
    neigh = KNeighborsClassifier(n_neighbors=3)
    # feat_p = []
    # feat_n = []
    # for idx, lal in enumerate(lals):
    #     if lal == 0:
    #         feat_n.append(feat_pca[idx,:])
    #     else:
    #         feat_p.append(feat_pca[idx,:])
    # print(len(feat_p), len(feat_n))
    neigh.fit(feat_pca, lals)
    return [pca, neigh]

def main(args):
    """ Main entry.
    """
    logging.info("Creating extractor ...")
    if 0:
        extractor = get_extractor(
            args.extractor, args.conf_file,
            {
                'use_gpu': args.gpu,
                'device_id': args.device_ids[0],
            })
    else:
        extractor = True
    logging.info("\tLoading Deep Model, Done!")

    if extractor is not None:
        wsi_root = '/home/dywang/CLC/Samples/wsi'
        hp_imgs = [l.strip().split('/')[-1] for l in os.popen('ls %s/T*062*.png'%args.input_folder)]

        logging.info("\tLoading Classification Model...")
        #classifier = load_classificer(args)
        classifier = None
        logging.info("\tLoading Classification Model, Done!")

        for hp_img in hp_imgs:
            hp_img_path = '%s/%s'%(args.input_folder, hp_img)
            hp_img_root = hp_img.split('.')[0]
            if hp_img[0] == 'T':
                wsi_path = '%s/01Tumor/%s.tif'%(wsi_root, hp_img_root)
            else:
                wsi_path = '%s/02Normal/%s.tif'%(wsi_root, hp_img_root)

            gen_heatmap_wsi(extractor, args.feat_name, classifier, wsi_path, hp_img_path, \
                args.model_level, args.mask_level, args.heatmap_level, args.window_size, args.augmentation, args.batch_size, args.group_size, args.step_size)
            break
if __name__ == '__main__':
    args = create_parser().parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
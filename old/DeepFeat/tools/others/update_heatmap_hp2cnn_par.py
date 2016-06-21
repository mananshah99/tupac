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
from extract_wsi_par import heatmap_generator_par

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

#
# split list into groups
#
def list_split(l, n):
    return [ l[i:i+n] for i in range(0, len(l), n) ]

def my_range(idx1, idx2, step):
    ids = range(idx1, idx2+1, step)
    if ids[-1] != idx2:
        ids.append(idx2)
    return ids

def getAFolder(folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    return folderName

class update_heatmap_par(heatmap_generator_par):
    def __init__(self, args, extractor):
        heatmap_generator_par.__init__(self, args, extractor)
        self.threshold = args.threshold
    # ml -> model level 0, 1, 2, 3, 4, 5
    # kl -> mask level 5
    # hl -> heatmap level 2
    def gen_postion_list(self):
        wsi_list, msk_list = [], []
        for line in [l.strip() for l in open(self.input_list) if l[0] != '#' and len(l.strip())>0]:
            itms = line.strip().split()
            wsi_list.append('%s/%s'%(self.input_folder, itms[0]))
            msk_list.append('%s/%s'%(self.input_folder, itms[1]))

        for wsiName, mskName in zip(wsi_list, msk_list):
            wsiNameRoot = wsiName.split('/')[-1].split('.')[0]
            sub_output_folder = getAFolder('%s/lst/%s'%(self.output_folder, wsiNameRoot))

            wsi = osi.open_slide(wsiName)
            heatmap_ori = ski.img_as_float(skio.imread(mskName, True))

            img_w_ml, img_h_ml = wsi.level_dimensions[self.deep_model_level]
            img_w_hl, img_h_hl = wsi.level_dimensions[self.heatmap_level]

            # get sampleing points
            hc_hl_lst, wc_hl_lst = [], []
            label_roi = label(heatmap_ori > self.threshold)
            Rs = regionprops(label_roi)
            for R in Rs:
                min_row, min_col, max_row, max_col = R['bbox']
                for r in my_range(min_row, max_row, self.step_size):
                    for c in my_range(min_col, max_col, self.step_size):
                        hc_hl_lst.append(r)
                        wc_hl_lst.append(c)

            # mapping sampling points to Level 0
            inds_list = []
            for hc_hl, wc_hl in zip(hc_hl_lst, wc_hl_lst):
                hc_ml, wc_ml = hc_hl * np.power(2, self.heatmap_level - self.deep_model_level), wc_hl * np.power(2, self.heatmap_level - self.deep_model_level)
                h1_ml, w1_ml = hc_ml - self.window_size / 2, wc_ml - self.window_size / 2
                inds_list.append([h1_ml, w1_ml])
            logging.info("\t %s : %d patches in total!"%(wsiName, len(inds_list)))
            inds_list_group = list_split(inds_list, self.group_size)

            for gidx, inds in enumerate(inds_list_group):
                with open('%s/%s_%05d.lst'%(sub_output_folder, wsiNameRoot, gidx), 'w') as f:
                    for h1_ml, w1_ml in inds:
                        f.write('%s %d %d\n'%(wsiName, h1_ml, w1_ml))

    def merge_result(self):
        heatmap_root_folder = getAFolder('%s/heatmap'%(self.output_folder))
        wsi_list, msk_list = [], []
        for line in [l.strip() for l in open(self.input_list) if l[0] != '#' and len(l.strip())>0]:
            itms = line.strip().split()
            wsi_list.append('%s/%s'%(self.input_folder, itms[0]))
            msk_list.append('%s/%s'%(self.input_folder, itms[1]))

        for wsiName, mskName in zip(wsi_list, msk_list):
            outputName = '%s/%s'%(heatmap_root_folder, wsiName.split('/')[-1].split('.')[0] + '.png')
            if os.path.exists(outputName):
                print("Omit:%s"%(outputName))
            else:
                print("Generating:%s"%(outputName))
                wsi = osi.open_slide(wsiName)

                img_w_ml, img_h_ml = wsi.level_dimensions[self.deep_model_level]
                img_w_hl, img_h_hl = wsi.level_dimensions[self.heatmap_level]

                # get all values
                if 1:
                    wsiNameRoot = wsiName.split('/')[-1].split('.')[0]
                    lst_output_folder = '%s/lst/%s'%(self.output_folder, wsiNameRoot)
                    lst_files = [l.strip() for l in os.popen('ls %s'%(lst_output_folder))]
                    inds_list, values = [], []
                    for lst_file in lst_files:
                        lst_file_path = '%s/lst/%s/%s'%(self.output_folder, wsiNameRoot, lst_file)
                        val_file_path = '%s/values/%s/%s.npy'%(self.output_folder, wsiNameRoot, lst_file.split('.')[0])
                        lst_info = [l.strip().split() for l in open(lst_file_path)]
                        value = [v for v in np.load(val_file_path)]
                        ind_list = [(int(i[1]), int(i[2])) for i in lst_info]
                        ##
                        inds_list += ind_list
                        values += value

                # genearte the heatmap
                heatmap = np.zeros((img_h_hl, img_w_hl, 2))
                if 1:
                    logging.info("\t Updating heatmap...")
                    for (h1_ml, w1_ml), v in zip(inds_list, values):
                        h1_hl = int(h1_ml / np.power(2, self.heatmap_level - self.deep_model_level))
                        w1_hl = int(w1_ml / np.power(2, self.heatmap_level - self.deep_model_level))
                        window_size_hl = int(self.window_size / np.power(2, self.heatmap_level - self.deep_model_level))

                        ori_patch = heatmap[h1_hl:h1_hl+window_size_hl, w1_hl:w1_hl+window_size_hl, :]
                        l1, c1 = np.split(ori_patch, 2, axis=2)

                        if l1.shape[0] > 0: # otherwise, the patches are out of bound
                            nmask = np.tile(np.array([v, 1]).reshape((1,1,2)), (c1.shape[0], c1.shape[1], 1))
                            l2, c2 = np.split(nmask, 2, axis=2)
                            ## merging
                            c3 = c1 + c2
                            l3 = ((l1 * c1) + l2) / c3
                            heatmap[h1_hl:h1_hl+window_size_hl, w1_hl:w1_hl+window_size_hl, :] = np.dstack((l3, c3))
                    logging.info("\t Done!")

                    heatmap = np.squeeze(heatmap[:,:,0])
                    heatmap_n = heatmap
                if 1:
                    heatmap_n = (heatmap_n * 255).astype(np.uint)
                    skio.imsave(outputName, heatmap_n)

    def run(self):
        if self.input_type == 11: # generate tasks
            self.gen_postion_list()
        elif self.input_type == 12:
            self.gen_task()
        elif self.input_type == 13:
            self.merge_result()
        elif self.input_type == 2: # a single patch
            self.run_single_task()
        else:
            print("ERROR input type")

def create_parser():
    """ Parse program arguments.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=
                                     argparse.RawTextHelpFormatter)
    parser.add_argument('extractor', type=str, help='name of extractor')
    parser.add_argument('conf_file', type=str, help='config file')
    parser.add_argument('feat_name', type=str, help='feature name')
    parser.add_argument('input_folder', type=str, help='output folder')
    parser.add_argument('input_list', type=str, help='input list: wsi&mask list / wsi&coordiate list')
    parser.add_argument('input_type', type=int, help='input type: 11 12 13 2')
    parser.add_argument('deep_model_level', type=int, help='deep model level')
    parser.add_argument('heatmap_level', type=int, help='the level of heatmap')
    parser.add_argument('mask_image_level', type=int, help='mask image level')
    parser.add_argument('threshold', type=float, help='threshold value')
    parser.add_argument('output_folder', type=str, help='output folder')

    parser.add_argument('--augmentation', type=int, default=1, help='do rotation')
    parser.add_argument('--window_size', type=int, default=256, help='windows size')
    parser.add_argument('--step_size', type=int, default=128, help='step size')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--group_size', type=int, default=100, help='group size')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0], help='config file')
    parser.add_argument('--gpu', action='store_true', help='config file')
    parser.add_argument("--log", type=str, default="INFO", help="log level")

    return parser

def main(args):
    """ Main entry.
    """
    logging.info("Creating extractor ...")
    extractor = get_extractor(
        args.extractor, args.conf_file,
        {
            'use_gpu': args.gpu,
            'device_id': args.device_ids[0],
        })
    logging.info("\tDone!")

    heatmap_level = max(args.deep_model_level, args.heatmap_level)
    if extractor is not None:
        hpgen = update_heatmap_par(args, extractor)
        hpgen.run()

if __name__ == '__main__':
    args = create_parser().parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)

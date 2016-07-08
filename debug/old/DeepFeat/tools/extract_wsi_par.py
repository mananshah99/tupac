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
from tqdm import tqdm
import skimage.io as skio

def save_sparse_matrix(outputName, data):
    m = sparse.csr_matrix(data)
    io.mmwrite(outputName, m)

def load_sparse_matrix(inputName):
    newm = io.mmread(inputName)
    return newm.toarray()

#
# split list into groups
#
def list_split(l, n):
    return [ l[i:i+n] for i in range(0, len(l), n) ]

def getAFolder(folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    return folderName

class heatmap_generator_par:
    def __init__(self, args, extractor):
        self.extractor = extractor
        self.input_folder = args.input_folder
        self.input_list = args.input_list
        self.input_type = args.input_type
        self.args = args
        ##
        self.feat_name = args.feat_name
        self.mask_image_level = args.mask_image_level
        self.deep_model_level = args.deep_model_level
        self.heatmap_level = args.heatmap_level
        self.window_size = args.window_size
        self.augmentation = args.augmentation
        self.batch_size = args.batch_size
        self.group_size = args.group_size
        self.step_size = args.step_size
        self.output_folder = args.output_folder
    #
    # Get prediction results using DL
    # len(v_img) = augmentation * num_patches
    #
    def doExtraction(self, v_img, batch_size):
        values_list = []
        v_img_list = list_split(v_img, batch_size)
        values = []
        for i, v_img_i in enumerate(v_img_list):
            logging.info("\t\t\t sub-group %d (%d)"%(i, len(v_img_i)))
            values_i = self.extractor.batch_extract_numpy(v_img_i, [self.feat_name])
            values_i = values_i[0] # get the first feature group
            values_i = values_i[:, 1].reshape(-1) # get the possibility of positive
            values_i[values_i<1e-4] = 0.0
            values = np.concatenate((values, values_i))
        values = values.reshape(-1, self.augmentation) # numpy is row first
        values_m = np.mean(values, axis=1)
        values_m = values_m.reshape(-1)
        return values_m
    #
    # data augmentation
    #
    def image_augmentation(self, img_000):
        if self.augmentation == 1:
            return [img_000]
        elif self.augmentation == 6:
            img_090 = nd.rotate(img_000, 90)
            img_180 = nd.rotate(img_090, 180)
            img_270 = nd.rotate(img_180, 270)
            img_fph = np.fliplr(img_000)
            img_fpv = np.flipud(img_000)
            return [img_000, img_090, img_180, img_270, img_fph, img_fpv]
        else:
            print("ERROR: Invalid Augmentation Number, reset it to 1!")
            self.augmentation = 1
            return [img_0000]
    #
    # extract image patches from WSI on LEVEL_0 using the
    #
    def get_image(self, wsi, h1_ml, w1_ml):
        h1_ml_2_level0 = h1_ml * np.power(2, self.deep_model_level)
        w1_ml_2_level0 = w1_ml * np.power(2, self.deep_model_level)
        img = wsi.read_region(
                (w1_ml_2_level0, h1_ml_2_level0),
                self.deep_model_level,
                (self.window_size, self.window_size)
                )
        img = np.asarray(img)[:,:,:3]
        imgs = self.image_augmentation(img)
        return imgs
    #
    # check the overlapping with tissue segmentation, which is helpful to reduce
    # the number of pataches
    #
    def has_overlapping_with_tissue_using_mask(self, msk, h1_ml, w1_ml, gap_between_mask_model):
        if msk is None:
            return True
        else:
            h1_kl = int(h1_ml / np.power(2, gap_between_mask_model))
            w1_kl = int(w1_ml / np.power(2, gap_between_mask_model))
            window_size_kl = int(self.window_size / np.power(2, gap_between_mask_model))
            return np.sum(msk[h1_kl:h1_kl + window_size_kl, w1_kl:w1_kl + window_size_kl]) > 0

    # ml -> model level 0, 1, 2, 3, 4, 5
    # kl -> mask level 5
    # hl -> heatmap level 2
    def gen_postion_list(self):
        wsi_list, msk_list = [], []
        for line in [l.strip() for l in open(self.input_list) if l[0] != '#' and len(l.strip())>0]:
            itms = line.strip().split()
            if len(itms) == 2:
                wsi_list.append('%s/%s'%(self.input_folder, itms[0]))
                msk_list.append('%s/%s'%(self.input_folder, itms[1]))
            elif len(itms) == 1:
                wsi_list.append('%s/%s'%(self.input_folder, itms[0]))
                msk_list.append(None)

        for wsiName, mskName in zip(wsi_list, msk_list):
            wsiNameRoot = wsiName.split('/')[-1].split('.')[0]
            sub_output_folder = getAFolder('%s/lst/%s'%(self.output_folder, wsiNameRoot))

            wsi = osi.open_slide(wsiName)
            if mskName:
                msk = skio.imread(mskName, True)
            else:
                msk = None

            img_w_ml, img_h_ml = wsi.level_dimensions[self.deep_model_level]
            img_w_hl, img_h_hl = wsi.level_dimensions[self.heatmap_level]
            # generate all the patches locations
            inds_list = []
            for h1_ml in range(0, img_h_ml, self.step_size):
                for w1_ml in range(0, img_w_ml, self.step_size):
                    if h1_ml + self.window_size > img_h_ml:
                        h1_ml = img_h_ml - self.window_size
                    if w1_ml + self.window_size > img_w_ml:
                        w1_ml = img_w_ml - self.window_size
                    if self.has_overlapping_with_tissue_using_mask(msk, h1_ml, w1_ml, self.mask_image_level - self.deep_model_level):
                        inds_list.append([h1_ml, w1_ml])
            logging.info("\t %s : %d patches in total!"%(wsiName, len(inds_list)))
            inds_list_group = list_split(inds_list, self.group_size)

            for gidx, inds in enumerate(inds_list_group):
                with open('%s/%s_%05d.lst'%(sub_output_folder, wsiNameRoot, gidx), 'w') as f:
                    for h1_ml, w1_ml in inds:
                        f.write('%s %d %d\n'%(wsiName, h1_ml, w1_ml))

    def gen_cmd(self, input_list, input_type):
        input_list_name = input_list.split('/')[-1]
        getAFolder('%s/task'%(self.output_folder))
        getAFolder('%s/task/log'%(self.output_folder))
        cmd = []
        cmd.append("#!/bin/bash")
        cmd.append("#BSUB -W 02:00")
        cmd.append("#BSUB -q short")
        cmd.append("#BSUB -n 2")
        cmd.append("#BSUB -o log/%s.out"%input_list_name)
        cmd.append("#BSUB -e log/%s.err"%input_list_name)
        cmd.append("export OMP_NUM_THREADS=1")
        cmd.append("export MKL_NUM_THREADS=1")
        cmd.append("export MKL_DOMAIN_NUM_THREADS=1")

        cmd.append("cd /home/dw140/dl/DeepFeat")
        cmd.append(
            "python ./tools/extract_wsi_par.py %s %s %s %s %s %d %d %s --heatmap_level %d --mask_image_level %d --augmentation %d --window_size %d  --step_size %d --batch_size %d --group_size %d"%(
                    self.args.extractor,
                    self.args.conf_file,
                    self.args.feat_name,
                    self.args.input_folder,
                    input_list,
                    input_type,
                    self.args.deep_model_level,
                    self.args.output_folder,
                    self.args.heatmap_level,
                    self.args.mask_image_level,
                    self.args.augmentation,
                    self.args.window_size,
                    self.args.step_size,
                    self.args.batch_size,
                    self.args.group_size
                )
        )
        return '\n'.join(cmd)

    def gen_task(self):
        task_root_folder = getAFolder('%s/task'%(self.output_folder))
        task_file_folder = getAFolder('%s/task/subs'%(self.output_folder))
        sub_folders = [l.strip() for l in os.popen('ls %s/lst'%(self.output_folder))]
        bsubs = []
        for sub_folder in sub_folders:
            print("\t %s"%sub_folder)
            lsts = [l.strip() for l in os.popen('ls %s/lst/%s/*.lst'%(self.output_folder, sub_folder))]
            for lst in lsts:
                lstName = lst.split('/')[-1].split('.')[0]
                print("\t\t %s"%lst)

                bsub_name = '%s.sub'%(lstName)
                cmd = self.gen_cmd(lst, 2)
                with open('%s/%s'%(task_file_folder, bsub_name), 'w') as f:
                    f.write(cmd)
                bsubs.append(bsub_name)
        with open('%s/runAll.sh'%(task_root_folder), 'w') as f:
            for bsub in bsubs:
                f.write('bsub < subs/%s\n'%bsub)

    def merge_result(self):
        heatmap_root_folder = getAFolder('%s/heatmap'%(self.output_folder))

        wsi_list, msk_list = [], []
        for line in [l.strip() for l in open(self.input_list) if l[0] != '#' and len(l.strip())>0]:
            itms = line.strip().split()
            if len(itms) == 2:
                wsi_list.append('%s/%s'%(self.input_folder, itms[0]))
                msk_list.append('%s/%s'%(self.input_folder, itms[1]))
            elif len(itms) == 1:
                wsi_list.append('%s/%s'%(self.input_folder, itms[0]))
                msk_list.append(None)

        for wsiName, mskName in zip(wsi_list, msk_list):
            wsi_name_root = wsiName.split('/')[-1].split('.')[0]
            outputName='%s/%s.png'%(heatmap_root_folder, wsi_name_root)
            if os.path.exists(outputName):
                print("Omit:%s"%(outputName))
            else:
                print("Generating:%s"%(outputName))
                wsi = osi.open_slide(wsiName)

                #img_w_ml, img_h_ml = wsi.level_dimensions[deep_model_level]
                img_w_hl, img_h_hl = wsi.level_dimensions[self.heatmap_level]
                heatmap = np.zeros((img_h_hl, img_w_hl, 2))
                # load inds and values
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
                if 1:
                    logging.info("\t Updating heatmap...")
                    inds_list_size = len(inds_list)
                    for (h1_ml, w1_ml), v in tqdm(zip(inds_list, values)):
                        h1_hl = int(h1_ml / np.power(2, self.heatmap_level - self.deep_model_level))
                        w1_hl = int(w1_ml / np.power(2, self.heatmap_level - self.deep_model_level))
                        window_size_hl = int(self.window_size / np.power(2, self.heatmap_level - self.deep_model_level))

                        ori_patch = heatmap[h1_hl:h1_hl+window_size_hl, w1_hl:w1_hl+window_size_hl, :]
                        l1, c1 = np.split(ori_patch, 2, axis=2)

                        nmask = np.tile(np.array([v, 1]).reshape((1,1,2)), (window_size_hl, window_size_hl, 1))
                        l2, c2 = np.split(nmask, 2, axis=2)

                        ## merging
                        c3 = c1 + c2
                        l3 = ((l1 * c1) + l2) / c3
                        heatmap[h1_hl:h1_hl+window_size_hl, w1_hl:w1_hl+window_size_hl, :] = np.dstack((l3, c3))
                        #break
                    logging.info("\t Done!")
                # Save the results
                if 1:
                    logging.info("\t Saving Mask...")
                    ## save the npy file ##
                    heatmap = np.squeeze(heatmap[:,:,0]).astype(np.float32)
                    heatmap[heatmap<1e-5] = 0.0
                    #np.save(outputName + ".npy", heatmap)

                    ## save the heat map ##
                    img_h_kl = img_h_hl / np.power(2, self.mask_image_level - self.heatmap_level)
                    img_w_kl = img_w_hl / np.power(2, self.mask_image_level - self.heatmap_level)
                    heatmap_kl = (resize(heatmap, (img_h_kl, img_w_kl)) * 255).astype(np.uint)
                    skio.imsave(outputName, heatmap_kl)

    def run_single_task(self):
        inds_list = []
        listNameRoot = args.input_list.split('/')[-1].split('.')[0]

        for line in [l.strip() for l in open(args.input_list)]:
            wsiName, h1_ml, w1_ml = line.strip().split()
            inds_list.append([int(h1_ml), int(w1_ml)])
        wsi = osi.open_slide(wsiName)
        wsiNameRoot = wsiName.split('/')[-1].split('.')[0]
        value_result_folder = getAFolder('%s/values/%s'%(self.output_folder, wsiNameRoot))
        result_name = '%s/%s.npy'%(value_result_folder, listNameRoot)
        if not os.path.exists(result_name):
            values = np.zeros((len(inds_list),), np.float32)
            inds_list_group = list_split(inds_list, self.group_size)
            for i, inds_list_group_i in enumerate(inds_list_group):
                v_img = []
                logging.info("\t\t Loading images ... : group %d:%d (%d) : %s"%(i, len(v_img), len(inds_list_group), wsiName))
                for h1_ml, w1_ml in inds_list_group_i:
                    imgs = self.get_image(wsi, h1_ml, w1_ml)
                    v_img += imgs
                logging.info("\t\t Make Prediction ... : group %d:%d (%d) : %s"%(i, len(v_img), len(inds_list_group), wsiName))
                values_i = self.doExtraction(v_img, self.batch_size)
                #values_i = np.zeros((len(v_img)))
                values[i*self.group_size:i*self.group_size + len(inds_list_group_i)] = values_i
            np.save(result_name, values)

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
    parser.add_argument('input_type', type=int, help='input type: 1 / 2')
    parser.add_argument('deep_model_level', type=int, help='deep model level')
    parser.add_argument('output_folder', type=str, help='output folder')

    parser.add_argument('--heatmap_level', type=int, default=2, help='the level of heatmap')
    parser.add_argument('--mask_image_level', type=int, default=5, help='mask image level')
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
        hpgen = heatmap_generator_par(args, extractor)
        hpgen.run()

if __name__ == '__main__':
    args = create_parser().parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)

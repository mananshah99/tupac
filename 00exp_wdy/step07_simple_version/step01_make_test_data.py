#!/usr/bin/env python
# coding: utf-8
#
# Copyright © 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.

import os,sys,argparse
import logging
import numpy as np
import skimage.io as skio
import openslide as osi

DESCRIPTION = """
"""

#img_fd = '/home/dywang/ServerDrive/Orchestra/Proliferation/data/TrainingData/training_image_data'
#msk_fd = '/data/dywang/Database/Proliferation/libs/00exp_wdy/stage02_getHP/wsi_msk'

img_fd = '/home/dywang/ServerDrive/Orchestra/Proliferation/data/TestingData'
msk_fd = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1-test_07-11-16'

def extend_inds_to_level0(h, w, args):
    gap = args.input_level - 0
    v = np.power(args.level_ratio, gap)
    hlist = h * v + np.arange(v)
    wlist = w * v + np.arange(v)
    hw = []
    for hv in hlist:
        for wv in wlist:
            hw.append([hv, wv])
    return hw

def get_tl_pts_in_level0(h_level0, w_level0, args):
    scale = np.power(args.level_ratio, args.out_level)
    wsize_level0 = args.wsize * scale
    wsize_level0_half = int(wsize_level0 / 2.0)

    h1_level0, w1_level0 = h_level0 - wsize_level0_half, w_level0 - wsize_level0_half
    return int(h1_level0), int(w1_level0)

def get_image(wsi, h1_level0, w1_level0, args):
    img = wsi.read_region(
            (w1_level0, h1_level0),
            args.out_level,
            (args.wsize, args.wsize))
    img = np.asarray(img)[:,:,:3]
    return img

def extract_patches(param):
    wsi_path, wsi_name, msk_path, args = param
    if os.path.exists(wsi_path) and os.path.exists(msk_path):
        print "wsi:", wsi_path
        wsi = osi.OpenSlide(wsi_path)
        msk = skio.imread(msk_path, True) / 255.0
        print wsi.dimensions
        print "msk:", msk_path, msk.shape

        ind_h, ind_w = np.nonzero(msk > args.threshold)
        print "\t#sampls:", len(ind_h)

        rinds = np.random.permutation(len(ind_h))
        ind_h = ind_h[rinds]
        ind_w = ind_w[rinds]
        print ind_h[:20]
        print ind_w[:20]
        ind_h_size = len(ind_h)
        ct = 0
        for patch_idx, (ch, cw) in enumerate(zip(ind_h, ind_w)):
            chcw_level0_list = extend_inds_to_level0(ch, cw, args)
            idx = int(len(chcw_level0_list) / 2.0)
            chcw_level0 = chcw_level0_list[idx]

            h_level0, w_level0 = chcw_level0
            print "\t\t %010d %010d -> %010d %010d"%(ch, cw, h_level0, w_level0)
            h1_level0, w1_level0 = get_tl_pts_in_level0(h_level0, w_level0, args)

            img = get_image(wsi, h1_level0, w1_level0, args)

            output_folder = '%s/%s'%(args.output_folder, wsi_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            img_name = '%s/%s_i%05d__h%010d_w%010d.png'%(output_folder, wsi_name, patch_idx, h1_level0, w1_level0)
            skio.imsave(img_name, img)
            print img_name
            ct = ct + 1
            if ct == args.number:
                print "\t\tDONE"
                break

def main(args):
    imgs, msks = [], []
    params = []
    for i in range(1, 321+1):
        img = '%s/TUPAC-TE-%03d.svs'%(img_fd, i)
        msk = '%s/TUPAC-TE-%03d.png'%(msk_fd, i)
        img_name = 'TUPAC-TE-%03d.svs'%(i)
        if os.path.exists(img) and os.path.exists(msk):
            params.append([img, img_name, msk, args])
    if 0:
        for param in params:
            extract_patches(param)
            #break
    else:
        from multiprocessing import Pool
        pool = Pool(10)
        pool.map(extract_patches, params)

def getargs():
    """ Parse program arguments.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--log", type=str, default="INFO", help="log level")
    parser.add_argument("--wsize", type=int, default=256, help='level ratio')
    parser.add_argument("--level_ratio", type=int, default=2, help='level ratio')
    parser.add_argument("--threshold", type=float, default=0, help="threshold")
    parser.add_argument("--number", type=int, default=1, help="level ratio")
    parser.add_argument("--input_level", type=int, default=0, help="level ratio")
    parser.add_argument("--out_level", type=int, default=1, help="level ratio")
    parser.add_argument("--output_folder", type=str, default="",  help="output folder")
    return parser.parse_args()

if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=numeric_level)
    main(args)

#!/usr/bin/env python
# coding: utf-8
#
# Copyright © 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.
DESCRIPTION = """
"""

import os
import sys
import argparse
import logging

from scipy.io import savemat

from deepfeat import get_extractor
import deepfeat.util as dutil


def create_parser():
    """ Parse program arguments.
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=
                                     argparse.RawTextHelpFormatter)
    parser.add_argument('extractor', type=str, help='name of extractor')
    parser.add_argument('conf_file', type=str, help='config file')
    parser.add_argument('feat_name', type=str, help='feature name')
    parser.add_argument('img_dir', type=str, help='image dir')
    parser.add_argument('out_dir', type=str, help='output dir')
    parser.add_argument('image_list', type=str, help='image list file')
    parser.add_argument('first_idx', type=int, nargs='?', default=-1,
                        help='first image for feature extraction' +
                        '(from the beginning if set to -1)')
    parser.add_argument('last_idx', type=int, nargs='?', default=-1,
                        help='last image for feature extraction' +
                        '(to the end if set to -1)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='number of featues stored in a file')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0],
                        help='config file')
    parser.add_argument('--gpu', action='store_true',
                        help='config file')
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser


def init(args):
    logging.info("Args:\n\t" + str(args))

    logging.info("Loading image list ...")
    with open(args.image_list, 'r') as lstf:
        images = [line.strip().split(' ')[0] for line in lstf]
    logging.info("\tDone!")

    dutil.makedirs(args.out_dir)

    # calculate image range
    num_img = len(images)
    first_idx = max(0, args.first_idx)
    last_idx = args.last_idx + 1 if args.last_idx >= 0 else num_img
    last_idx = min(last_idx, num_img)

    return images, first_idx, last_idx


def load_image(imgpath):
    with open(imgpath, 'rb') as imgf:
        imgdat = imgf.read()
    return imgdat


def load_images(img_dir, images, first, last):
    logging.info("Loading images (%d~%d) ..." % (first, last-1))
    v_img = [load_image(os.path.join(img_dir, images[i]))
             for i in xrange(first, last)]
    logging.info("\tDone!")
    return v_img


def dump_feat(extractor, feat_name, v_img, outpath):
    logging.info("Extracting features ...")
    [feat] = extractor.batch_extract(v_img, [feat_name])
    logging.info("\tDone!")

    savemat(outpath, {'feat': feat})


def main(args):
    """ Main entry.
    """
    images, first_idx, last_idx = init(args)

    logging.info("Creating extractor ...")
    extractor = get_extractor(
        args.extractor, args.conf_file,
        {
            'use_gpu': args.gpu,
            'device_id': args.device_ids[0],
        })
    logging.info("\tDone!")

    for i in xrange(first_idx, last_idx, args.batch_size):
        i_end = min(i + args.batch_size, last_idx)
        v_img = load_images(args.img_dir, images, i, i_end)
        dump_feat(extractor, args.feat_name, v_img, os.path.join(args.out_dir, "feat_%010d.mat" % i))

if __name__ == '__main__':
    args = create_parser().parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)

#!/usr/bin/env python
# coding: utf-8

"""
   File Name: extract_par.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Fri Jul 24 20:21:30 2015 CST
"""
DESCRIPTION = """
"""

import os
import sys
import argparse
import logging

from multiprocessing import Process, Queue

from scipy.io import savemat

from deepfeat import get_extractor
import deepfeat.util as dutil

from extract import init, load_images, dump_feat
from extract import create_parser as base_create_parser

def create_parser():
    """ Parse program arguments.
    """

    parser = base_create_parser()
    parser.add_argument('--num_proc', type=int, default=1,
                        help='config file')
    return parser


def extract_proc(inp_queue, extractor_name, conf_file,
                 img_dir, out_dir, resource):

    logging.info("Creating extractor ...")
    extractor = get_extractor(extractor_name, conf_file, resource)
    logging.info("\tDone!")

    while True:
        inp = inp_queue.get()
        if inp is None:
            break
        first_idx, v_img = inp

        dump_feat(extractor, args.feat_name, v_img,
                  os.path.join(out_dir, "feat_%d.mat" % first_idx))


def main(args):
    """ Main entry.
    """
    images, first_idx, last_idx = init(args)

    v_dev = args.device_ids
    inp_queue = Queue(args.num_proc + 1)
    v_p = [
        Process(target=extract_proc,
                args=(inp_queue, args.extractor,
                      args.conf_file, args.img_dir, args.out_dir,
                      {
                          'use_gpu': args.gpu,
                          'device_id': v_dev[pid % len(v_dev)],
                      }))
        for pid in range(args.num_proc)]

    for p in v_p:
        p.start()

    for i in xrange(first_idx, last_idx, args.batch_size):
        i_end = min(i + args.batch_size, last_idx)
        v_img = load_images(args.img_dir, images, i, i_end)
        inp_queue.put((i, v_img))

    for pid in xrange(args.num_proc):
        inp_queue.put(None)

    for p in v_p:
        p.join()


if __name__ == '__main__':
    args = create_parser().parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)

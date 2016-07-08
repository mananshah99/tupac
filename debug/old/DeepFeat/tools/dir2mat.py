#!/usr/bin/env python
# coding: utf-8

"""
   File Name: tools/dir2mat.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Fri Sep 11 12:40:33 2015 CST
"""
DESCRIPTION = """
"""

import os
import argparse
import logging

from deepfeat.util import load_feat, save_feat


def runcmd(cmd):
    """ Run command.
    """

    logging.info("%s" % cmd)
    os.system(cmd)


def getargs():
    """ Parse program arguments.
    """

    parser = argparse.ArgumentParser(
            description=DESCRIPTION,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dir', type=str,
            help='dir containing matfiles')
    parser.add_argument('mat', type=str,
            help='single matfile')
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")

    return parser.parse_args()


def main(args):
    """ Main entry.
    """
    logging.info("Loading mat files ...")
    feat = load_feat(args.dir)
    logging.info("\tDone!")

    logging.info("Saving mat files ...")
    save_feat(args.mat, feat)
    logging.info("\tDone!")



if __name__ == '__main__':
    args = getargs()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)

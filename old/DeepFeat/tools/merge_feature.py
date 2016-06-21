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
import skimage.io as skio
import numpy as np
from random import sample
from skimage.transform import resize
from scipy import sparse, io
import scipy.ndimage as nd
from tqdm import tqdm
from sklearn.decomposition import PCA
import cPickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

def create_parser():
    """ Parse program arguments.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=
                                     argparse.RawTextHelpFormatter)
    parser.add_argument('feat_dir', type=str, help='name of extractor')
    parser.add_argument("--log", type=str, default="INFO",
                        help="log level")
    return parser

def main(args):
    """ Main entry.
    """
    mats = [l.strip() for l in os.popen('ls %s/mat/*.mat'%args.feat_dir)]
    nams = [l.strip().split()[0] for l in open('%s/lst'%args.feat_dir)]
    lals = [int(l.strip().split()[1]) for l in open('%s/lst'%args.feat_dir)]

    feat_list = []
    for mat in tqdm(mats):
        feat_i = np.squeeze(loadmat(mat)['feat'])
        feat_list.append(feat_i)
    feat = np.vstack(feat_list)

    pca_model_path = '%s/pca.pkl'%args.feat_dir
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
    print('fea shape', feat_pca.shape)
    neigh = KNeighborsClassifier(n_neighbors=100)

    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(feat_pca, lals, test_size=0.1, random_state=0)
    #neigh.fit(X_train, y_train)
    #s = neigh.score(X_test, y_test)
    #print(s)
    knn_model_path = '%s/knn.pkl'%args.feat_dir
    if not os.path.exists(knn_model_path):
        print("Train KNN...")
        neigh.fit(feat_pca, lals)
        with open(knn_model_path, 'wb') as fid:
            cPickle.dump(neigh, fid)

if __name__ == '__main__':
    args = create_parser().parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        level=numeric_level)
    main(args)
#!/usr/bin/env python
# coding: utf-8

"""
   File Name: caffe_extractor.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Thu Jul 23 20:40:19 2015 CST
"""
DESCRIPTION = """
"""

import os
import sys
import logging
import time
import ipdb
from deepfeat import Extractor
import numpy as np
import cv2
import util
#CAFFE_PYTHON = 'external/caffe/python'
#sys.path.append(CAFFE_PYTHON)
#import caffe

#----------------------- lasagne work ---- 

from theano import function, config, shared, sandbox

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer

try: 
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except:
    pass # assume that the user isn't trying to use lasagne

from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

class LasagneExtractor(Extractor):
    def __init__(self, caffe, conf_file, resource):
        super(LasagneExtractor, self).__init__(conf_file, resource)

        #prototxt = config.get('caffe', 'prototxt')
        #caffemodel = config.get('caffe', 'caffemodel')
        #meanv = [float(i.strip()) for i in config.get('caffe', 'meanv').split(',')]

        # parameters from the resource array (for example {'use_gpu': args.gpu, 'device_id':args.device_ids[0] })
        use_gpu = resource.get('use_gpu', False)
        device_id = resource.get('device_id', 0)

        # ipdb.set_trace()
        if use_gpu:
            # Theano will fall back to cpu if gpu does not work
            # Currently vi ~/.theanorc to see your config
            print("Using GPU") 
        else:
            print("Falling back to the CPU")

        print os.getcwd()
        
        import self.lasagnemodel # this is a python file that returns a network
        
        self.net = self.lasagnemodel.build_model() # build_model is the function that returns the network
        
        import pickle
        weights = pickle.load(open(self.lasagneweights))
        classes = weights['sysnet words']
        mean_image = weights['mean image']
        
        lasagne.layers.set_all_param_values(self.net, weights['values'])

    def transform(self, imgdat):
        img = cv2.imdecode(np.asarray(bytearray(imgdat), dtype=np.uint8),
                           cv2.CV_LOAD_IMAGE_COLOR).astype(np.float32) / 255.0
        return self.transformer.preprocess('data', img)

    def transform_numpy(self, img_numpy):
        r,g,b = cv2.split(img_numpy)
        img = cv2.merge((b,g,r)).astype(np.float32) / 255.0
        return self.transformer.preprocess('data', img)

    def extract(self, img, v_feat):
        self.net.blobs['data'].data[...] = self.transform(img)

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

    def batch_extract(self, v_img, v_feat):
        num_img = len(v_img)
        v_data = [self.transform(img) for img in v_img]

        n, c, h, w = self.net.blobs['data'].shape
        if n != num_img:
            self.net.blobs['data'].reshape(num_img, c, h, w)
        for i in xrange(num_img):
            self.net.blobs['data'].data[i, ...] = v_data[i]

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

    def batch_extract_numpy(self, v_img, v_feat):
        num_img = len(v_img)
        v_data = [self.transform_numpy(img) for img in v_img]

        n, c, h, w = self.net.blobs['data'].shape
        if n != num_img:
            self.net.blobs['data'].reshape(num_img, c, h, w)
        for i in xrange(num_img):
            self.net.blobs['data'].data[i, ...] = v_data[i]

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

class CaffeExtractor(Extractor):
    def __init__(self, caffe, conf_file, resource):
        super(CaffeExtractor, self).__init__(conf_file, resource)

        #prototxt = config.get('caffe', 'prototxt')
        #caffemodel = config.get('caffe', 'caffemodel')
        #meanv = [float(i.strip()) for i in config.get('caffe', 'meanv').split(',')]

        # parameters from the resource array (for example {'use_gpu': args.gpu, 'device_id':args.device_ids[0] })
        use_gpu = resource.get('use_gpu', False)
        device_id = resource.get('device_id', 0)

        # ipdb.set_trace()
        if use_gpu:
            caffe.set_device(device_id)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        print os.getcwd()

        # Initialize the CNN
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

        # Initialize caffe's online transformer
        self.transformer = caffe.io.Transformer(
            {'data': self.net.blobs['data'].data.shape}
        )

        self.transformer.set_transpose('data', (2, 0, 1)) # h w z - > z h w

        # mean pixel
        #mean = np.array([104.00698793,  116.66876762,  122.67891434])
        mean = np.array(self.meanv)
        self.transformer.set_mean('data', mean)

        # the model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)

        # # the reference model has channels in BGR order instead of RGB
        # self.transformer.set_channel_swap('data', (2,1,0))

        n, c, h, w = self.net.blobs['data'].shape
        self.net.blobs['data'].reshape(1, c, h, w)

    def transform(self, imgdat):
        img = cv2.imdecode(np.asarray(bytearray(imgdat), dtype=np.uint8),
                           cv2.CV_LOAD_IMAGE_COLOR).astype(np.float32) / 255.0
        return self.transformer.preprocess('data', img)

    def transform_numpy(self, img_numpy):
        r,g,b = cv2.split(img_numpy)
        img = cv2.merge((b,g,r)).astype(np.float32) / 255.0
        return self.transformer.preprocess('data', img)

    def extract(self, img, v_feat):
        self.net.blobs['data'].data[...] = self.transform(img)

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

    def batch_extract(self, v_img, v_feat):
        num_img = len(v_img)
        v_data = [self.transform(img) for img in v_img]

        n, c, h, w = self.net.blobs['data'].shape
        if n != num_img:
            self.net.blobs['data'].reshape(num_img, c, h, w)
        for i in xrange(num_img):
            self.net.blobs['data'].data[i, ...] = v_data[i]

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

    def batch_extract_numpy(self, v_img, v_feat):
        num_img = len(v_img)
        v_data = [self.transform_numpy(img) for img in v_img]

        n, c, h, w = self.net.blobs['data'].shape
        if n != num_img:
            self.net.blobs['data'].reshape(num_img, c, h, w)
        for i in xrange(num_img):
            self.net.blobs['data'].data[i, ...] = v_data[i]

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

class CaffeExtractor_HY(Extractor):
    def __init__(self, caffe, conf_file, resource):
        super(CaffeExtractor_HY, self).__init__(conf_file, resource)

        #prototxt = config.get('caffe', 'prototxt')
        #caffemodel = config.get('caffe', 'caffemodel')
        #meanv = [float(i.strip()) for i in config.get('caffe', 'meanv').split(',')]

        use_gpu = resource.get('use_gpu', False)
        device_id = resource.get('device_id', 0)

        self.net = caffe.Net(self.prototxt, self.caffemodel)
        if use_gpu:
            self.net.set_mode_gpu()
        else:
            self.net.set_mode_cpu()

        self.input_dim = self.net.blobs[self.net.inputs[0]].data.shape
        self.transformer = util.Transformer(
            {'data': self.input_dim}
        )

        self.transformer.set_transpose('data', (2, 0, 1)) # h w z - > z h w

        mean = np.array(self.meanv)
        self.transformer.set_mean('data', mean)
        self.transformer.set_raw_scale('data', 255)

        n, c, h, w = self.input_dim
        self.net.blobs['data'].reshape(1, c, h, w)

    def transform(self, imgdat):
        img = cv2.imdecode(np.asarray(bytearray(imgdat), dtype=np.uint8),
                           cv2.CV_LOAD_IMAGE_COLOR).astype(np.float32) / 255.0
        return self.transformer.preprocess('data', img)

    def transform_numpy(self, img_numpy):
        r,g,b = cv2.split(img_numpy)
        img = cv2.merge((b,g,r)).astype(np.float32) / 255.0
        return self.transformer.preprocess('data', img)

    def extract(self, img, v_feat):
        self.net.blobs['data'].data[...] = self.transform(img)

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

    def batch_extract(self, v_img, v_feat):
        num_img = len(v_img)
        v_data = [self.transform(img) for img in v_img]

        n, c, h, w = self.input_dim
        if n != num_img:
            self.net.blobs['data'].reshape(num_img, c, h, w)
        for i in xrange(num_img):
            self.net.blobs['data'].data[i, ...] = v_data[i]

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

    def batch_extract_numpy(self, v_img, v_feat):
        num_img = len(v_img)
        v_data = [self.transform_numpy(img) for img in v_img]

        n, c, h, w = self.input_dim
        if n != num_img:
            self.net.blobs['data'].reshape(num_img, c, h, w)
        for i in xrange(num_img):
            self.net.blobs['data'].data[i, ...] = v_data[i]

        logging.debug("Forwarding ...")
        start = time.time()
        self.net.forward()
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [self.net.blobs[feat].data for feat in v_feat]

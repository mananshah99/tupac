#!/usr/bin/env python
# coding: utf-8

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

from theano import function, config, shared, sandbox
import theano

import lasagne
from lasagne.utils import floatX

class LasagneExtractor(Extractor):
    def __init__(self, conf_file, resource):
        super(LasagneExtractor, self).__init__(conf_file, resource)

        #meanv = [float(i.strip()) for i in config.get('lasagne', 'meanv').split(',')]

        # parameters from the resource array (for example {'use_gpu': args.gpu, 'device_id':args.device_ids[0] })
        use_gpu = resource.get('use_gpu', False)
        device_id = resource.get('device_id', 0)

        # ipdb.set_trace()
        if use_gpu:
            # Theano will fall back to cpu if gpu does not work
            # vi ~/.theanorc to see your config
            print("Using GPU") 
        else:
            print("Falling back to the CPU")

        print os.getcwd()

        variables = {}
        execfile(self.lasagnemodel, variables)
        self.net = variables['NETWORK']  # build_model is the function that returns the network
        
        import pickle
        weights = pickle.load(open(self.lasagneweights))
        
        lasagne.layers.set_all_param_values(self.net, weights['param values'])

    def transform(self, imgdat):
        img = cv2.imdecode(np.asarray(bytearray(imgdat), dtype=np.uint8),
                           cv2.CV_LOAD_IMAGE_COLOR).astype(np.float32) / 255.0
        return img # self.transformer.preprocess('data', img)

    def transform_numpy(self, img_numpy):
        r,g,b = cv2.split(img_numpy)
        img = cv2.merge((b,g,r)).astype(np.float32) / 255.0
        def preprocess_image(image):
            # for lasagne w/o Caffe's tansformer
            image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)
            image = image[::-1, :, :] # switch to BGR
            # if we have mean values image -= MEAN_VALUE (array for example is MEAN_VALUE = np.array([103.939, 116.779, 123.68]))
            return image

        return preprocess_image(img) # self.transformer.preprocess('data', img)

    def extract(self, img, v_feat):
        raise NotImplementedError('Not needed in extract_wsi_tupac.py')

    def batch_extract(self, v_img, v_feat):
        raise NotImplementedError('transform(...) has not yet been fully implemented')

    def batch_extract_numpy(self, v_img, v_feat):
        num_img = len(v_img)
        v_data = [self.transform_numpy(img) for img in v_img]

        # number of images, three color channels, WINDOW_SIZE, WINDOW_SIZE
        patch_arr = np.zeros((num_img, 3, 256, 256), np.float32)

        for i in xrange(num_img):
            patch_arr[i, ...] = v_data[i]
        
        logging.debug("Patch array shape: ", patch_arr.shape)
        logging.debug("Forwarding ...")
        start = time.time()
        prob = np.array(lasagne.layers.get_output(self.net, patch_arr, deterministic=True).eval())
        
        # Rationale behind the next few lines of code:
        # --------------------------------------------
        # At the end of the day, the network output will be a set of two probabilities => [p1, p2]
        # These probabilities are 1) the probability of mitosis and 2) the probability of not mitosis
        # As the code currently uses googlenet trained on ImageNet, it returns a set of 1k probabilities
        # for the different classes present in ImageNet. For testing purposes, we extract the first two of 
        # these probabilities and add them to the probabilities_trimmed_for_googlenet list. We then return
        # this list of probabilities as a [np.array]
   
        prob_trimmed_for_googlenet = []
        
        for x in prob:
            prob_trimmed_for_googlenet.append([x[0], x[1]]) # this value is meaningless but at least it's a probability 
       
        logging.debug(np.array(prob_trimmed_for_googlenet))
 
        logging.debug("\tDone! (%.4fs)" % (time.time() - start))
        return [np.array(prob_trimmed_for_googlenet, dtype=np.float32)] 

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
        img = cv2.resize(img, (256, 256)) # for googlenet input we need 256 x 256
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

        print ">>>> NUM_IMG ", num_img
        print ">>>> IMG SIZE ", v_data[0].shape

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

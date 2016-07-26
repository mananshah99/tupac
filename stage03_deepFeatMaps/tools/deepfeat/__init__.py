#!/usr/bin/env python
# coding: utf-8

"""
   File Name: __init__.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Fri Jul 24 19:07:42 2015 CST
"""
DESCRIPTION = """
"""

import logging, util, sys
import ConfigParser

class Extractor(object):
    def update_root(self, root, input_path):
        if input_path[0] != '/':
            input_path = '%s/%s'%(root, input_path)
        return input_path

    def __init__(self, conf_file, resource):
        self.resource = resource
        conf_file_path = '/'.join(conf_file.split('/')[:-1])
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        
        try:
            model = config.get('lasagne', 'model')
            weights = config.get('lasagne', 'weights')
            self.lasagnemodel = self.update_root(conf_file_path, model)
            self.lasagneweights = self.update_root(conf_file_path, weights) 
            print("[OK] Lasagne config loaded")
        except:
            print("[!] Lasagne config not loaded")

        try:
            prototxt = config.get('caffe', 'prototxt')
            caffemodel = config.get('caffe', 'caffemodel')
        
            self.prototxt = self.update_root(conf_file_path, prototxt)
            self.caffemodel = self.update_root(conf_file_path, caffemodel)
            self.meanv = [float(i.strip()) for i in config.get('caffe', 'meanv').split(',')]
            print("[OK] Caffe config loaded")
        except Exception as e:
            print e
            print("[!] Caffe config not loaded")

def get_extractor(name, conf_file, resource):
    logging.info("Creating `%s` feature extractor ..." % name)
    
    if name == 'caffe':
      CAFFE_PYTHON = 'external/caffe/python'
      from caffe_extractor import CaffeExtractor as caffeFeature

    elif name == 'caffe-no-resize':
      CAFFE_PYTHON = 'external/caffe/python'
      from caffe_extractor_full import CaffeExtractor as caffeFeature
 
    elif name == 'caffe_hy':
      from caffe_extractor import CaffeExtractor_HY as caffeFeature
      CAFFE_PYTHON = 'external/caffe_hy/python'

    elif name == 'lasagne':
        from caffe_extractor import LasagneExtractor as lasagneFeature
    
    else:
      logging.error("Failed to set CAFFE_PYTHON")
      return None
   
    if name == 'caffe' or name == 'caffe-no-resize' or name == 'caffe_hy': 
        sys.path.append(CAFFE_PYTHON)
        logging.info("CAFFE_PYTHON: %s"%(CAFFE_PYTHON))
    
        import caffe
    
        extractor = caffeFeature(caffe, conf_file, resource)
        logging.info("`%s` feature extractor created" % name)
        return extractor
    
    elif name == 'lasagne':
        extractor = lasagneFeature(conf_file, resource)
        logging.info("`%s` feature extractor created" % name)
    
    return extractor 


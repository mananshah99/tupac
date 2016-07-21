#!/usr/bin/env python
# coding: utf-8
#
# Copyright © 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.

import os,sys,argparse
import logging


def gen_initial_list(ROOT, wsi_list, output_name):
    lines_all = []
    for fd, label in [['pos', 1], ['neg', 0]]:
        lines = []
        for wsi_name in wsi_list:
            print 'wsi_name: ', wsi_name
            hpf_names = [l.strip() for l in os.popen('ls %s/%s/%s'%(ROOT, fd, wsi_name))]
            for hpf_name in hpf_names:
                print "\thpf_name: ", hpf_name
                imgs = [l.strip() for l in os.popen('ls %s/%s/%s/%s'%(ROOT, fd, wsi_name, hpf_name))]
                print '\t\t#img=', len(imgs)
                for img in imgs:
                    lines.append('%s/%s/%s/%s %d'%(fd, wsi_name, hpf_name, img, label))
        lines_all += lines
    print "%s -> #img = %d"%(fd, len(lines))
    with open(output_name, 'w') as f:
        f.write('\n'.join(lines_all))

ROOT = 'training_examples_s2'
wsi_tr_list = ['%02d'%i for i in range(1, 13, 1)]
wsi_te_list = ['%02d'%i for i in range(13, 24, 1)]
print wsi_tr_list
print wsi_te_list
gen_initial_list(ROOT, wsi_tr_list, 'mc13_tr.lst')
gen_initial_list(ROOT, wsi_te_list, 'mc13_te.lst')
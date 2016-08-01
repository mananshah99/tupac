#!/usr/bin/env python
# coding: utf-8
#
# Copyright © 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.

import os,sys,argparse
import logging


def gen_initial_list(ROOT, prefix):
    fdlist = ['pos', 'neg']
    # draw ground truth
    for fd in fdlist:
        lines = []
        wsi_names = [l.strip() for l in os.popen('ls %s/%s'%(ROOT, fd))]
        for wsi_name in wsi_names:
            print 'wsi_name: ', wsi_name
            hpf_names = [l.strip() for l in os.popen('ls %s/%s/%s'%(ROOT, fd, wsi_name))]
            for hpf_name in hpf_names:
                print "\thpf_name: ", hpf_name
                imgs = [l.strip() for l in os.popen('ls %s/%s/%s/%s'%(ROOT, fd, wsi_name, hpf_name))]
                print '\t\t#img=', len(imgs)
                for img in imgs:
                    lines.append('%s/%s/%s/%s'%(fd, wsi_name, hpf_name, img))
        print "%s -> #img = %d"%(fd, len(lines))
        with open('%s/%s_%s.lst'%(ROOT, prefix, fd), 'w') as f:
            f.write('\n'.join(lines))

def gen_train_data_stage_01(ROOT, prefix):
    p = ['%s 1'%(l.strip()) for l in open('%s/%s_pos.lst'%(ROOT, prefix))]
    n = ['%s 0'%(l.strip()) for l in open('%s/%s_neg.lst'%(ROOT, prefix))]
    import random

    r = 0.9
    p_s, n_s = len(p), len(n)
    random.shuffle(p)
    random.shuffle(n)

    p_tr_s = int(p_s * r)
    n_tr_s = int(n_s * r)

    with open('%s/list_%s_tr.lst'%(ROOT, prefix), 'w') as f:
        f.write('\n'.join(p[:p_tr_s]))
        f.write('\n')
        f.write('\n'.join(n[:n_tr_s]))

    with open('%s/list_%s_te.lst'%(ROOT, prefix), 'w') as f:
        f.write('\n'.join(p[p_tr_s+1:]))
        f.write('\n')
        f.write('\n'.join(n[n_tr_s+1:]))

ROOT = 'training_examples'
gen_initial_list(ROOT, 'stage1')
gen_train_data_stage_01(ROOT, 'stage1')

#ROOT = 'training_examples_s2'
#gen_initial_list(ROOT, 'stage2')
#gen_train_data_stage_01('stage2')

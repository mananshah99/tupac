#!/usr/bin/env python
# coding: utf-8
#
# Copyright © 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.

import os,sys,argparse
import logging


def gen_initial_list(ROOT, prefix, fdlist = ['pos', 'neg']):
    #fdlist = ['pos', 'neg']
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
    p = ['%s 1'%(l.strip()) for l in open('%s/stage1_pos.lst'%(ROOT))]
    n = ['%s 0'%(l.strip()) for l in open('%s/stage1_neg.lst'%(ROOT))]
    import random

    print "p, n", len(p), len(n)

    r = 0.9

    PN, np = 5, []
    for i in range(PN):
        np += p
    p = np

    p_s = len(p)
    n = n[:p_s]
    n_s = len(n)

    random.shuffle(p)
    random.shuffle(n)

    print "p, n", len(p), len(n)

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


def gen_train_data_stage_02(ROOT, prefix):
    import random

    p = ['%s 1'%(l.strip()) for l in open('%s/stage1_pos.lst'%(ROOT))]
    n1 = ['%s 0'%(l.strip()) for l in open('%s/stage1_neg.lst'%(ROOT))]
    n2 = ['%s 0'%(l.strip()) for l in open('%s/stage2_neg2.lst'%(ROOT))]

    n = n1 + n2

    p_ext = []
    for i in range(len(n) // len(p)):
        p_ext += p
    random.shuffle(p)
    p_ext += p[:len(n) - len(p_ext)]
    p = p_ext

    r = 0.9
    random.shuffle(p)
    random.shuffle(n)
    p_s, n_s = len(p), len(n)
    p_tr_s = int(p_s * r)
    n_tr_s = int(n_s * r)

    print len(p), len(n)
    tr_lines = p[:p_tr_s] + n[:n_tr_s]
    te_lines = p[p_tr_s+1:] + n[n_tr_s+1:]
    random.shuffle(tr_lines)
    random.shuffle(te_lines)

    with open('%s/list_%s_tr.lst'%(ROOT, prefix), 'w') as f:
        f.write('\n'.join(tr_lines))

    with open('%s/list_%s_te.lst'%(ROOT, prefix), 'w') as f:
        f.write('\n'.join(te_lines))

def gen_train_data_stage_02_top12(ROOT, prefix):
    import random
    NUM = 12

    p = ['%s 1'%(l.strip()) for l in open('%s/stage1_pos.lst'%(ROOT)) if int(l.strip().split('/')[1]) <=NUM ]
    n1 = ['%s 0'%(l.strip()) for l in open('%s/stage1_neg.lst'%(ROOT)) if int(l.strip().split('/')[1]) <=NUM ]
    n2 = ['%s 0'%(l.strip()) for l in open('%s/stage2_neg2.lst'%(ROOT)) if int(l.strip().split('/')[1]) <=NUM ]

    n = n1 + n2

    p_ext = []
    for i in range(len(n) // len(p)):
        p_ext += p
    random.shuffle(p)
    p_ext += p[:len(n) - len(p_ext)]
    p = p_ext

    r = 0.9
    random.shuffle(p)
    random.shuffle(n)
    p_s, n_s = len(p), len(n)
    p_tr_s = int(p_s * r)
    n_tr_s = int(n_s * r)

    print len(p), len(n)
    tr_lines = p[:p_tr_s] + n[:n_tr_s]
    te_lines = p[p_tr_s+1:] + n[n_tr_s+1:]
    random.shuffle(tr_lines)
    random.shuffle(te_lines)

    with open('%s/list_%s_tr_top12.lst'%(ROOT, prefix), 'w') as f:
        f.write('\n'.join(tr_lines))

    with open('%s/list_%s_te_top12.lst'%(ROOT, prefix), 'w') as f:
        f.write('\n'.join(te_lines))

ROOT = 'training_examples'
#gen_initial_list(ROOT, 'stage1')
#gen_train_data_stage_01(ROOT, 'stage1')

#gen_initial_list(ROOT, 'stage2', fdlist=['neg2'])
gen_train_data_stage_02(ROOT, 'stage2')
gen_train_data_stage_02_top12(ROOT, 'stage2')

#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools

GLOG_logtostderr=0 \
GLOG_alsologtostderr=1 \
GLOG_stderrthreshold=1 \
GLOG_log_dir=$LOG \
    $TOOLS/caffe test -model train_val-resultlayer-3stack.prototxt \
    -weights models-resultlayer_iter_64500.caffemodel \
    -iterations 10 \
    --gpu 1
    
#    -weights models.final/round01_gnet_iter_140000.caffemodel \

#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools

GLOG_logtostderr=0 \
GLOG_alsologtostderr=1 \
GLOG_stderrthreshold=1 \
GLOG_log_dir=$LOG \
    $TOOLS/caffe test -model test.prototxt \
    -weights models/cnn10_iter_217316.caffemodel \
    -iterations 100 \
    --gpu 2

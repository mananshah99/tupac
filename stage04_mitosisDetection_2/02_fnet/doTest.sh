#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools

GLOG_logtostderr=0 \
GLOG_alsologtostderr=1 \
GLOG_stderrthreshold=1 \
GLOG_log_dir=$LOG \
    $TOOLS/caffe test -model test.prototxt \
    -weights models.final/round02_cnn10_iter_190000.caffemodel \
    -iterations 100 \
    --gpu 0

#    -weights models.final/round02_cnn10_iter_95000.caffemodel \
#    -weights models.final/round01_face_iter_35000.caffemodel \

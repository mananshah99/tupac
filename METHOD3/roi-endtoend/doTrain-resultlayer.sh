#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools
LOG=log
MODEL=models

mkdir $LOG
mkdir $MODEL

GLOG_logtostderr=0 \
GLOG_alsologtostderr=1 \
GLOG_stderrthreshold=1 \
GLOG_log_dir=$LOG \
    $TOOLS/caffe train --solver=./solver-resultlayer.prototxt \
    --weights=models/cnn10_iter_217316.caffemodel  \
    --gpu=3

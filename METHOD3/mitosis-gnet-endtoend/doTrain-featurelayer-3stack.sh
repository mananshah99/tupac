#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools
LOG=log-featurelayer
MODEL=models-featurelayer

mkdir $LOG
mkdir $MODEL

GLOG_logtostderr=0 \
GLOG_alsologtostderr=1 \
GLOG_stderrthreshold=1 \
GLOG_log_dir=$LOG \
    $TOOLS/caffe train --solver=./solver-featurelayer-3stack.prototxt \
    --weights=cnn10_iter_23559.caffemodel  \
    --gpu=0,1,2

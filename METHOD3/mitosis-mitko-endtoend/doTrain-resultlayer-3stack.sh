#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools
LOG=log-resultlayer
MODEL=models-resultlayer

mkdir $LOG
mkdir $MODEL

GLOG_logtostderr=0 \
GLOG_alsologtostderr=1 \
GLOG_stderrthreshold=1 \
GLOG_log_dir=$LOG \
    $TOOLS/caffe train --solver=./solver-resultlayer-3stack.prototxt \
    --weights=models.final/mitosis_detection_wo_norm.caffemodel  \
    --gpu=1,2,3

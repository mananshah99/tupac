#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe-augmentation/build/tools
LOG=log
MODEL=models
mkdir $LOG
mkdir $MODEL

GLOG_logtostderr=0 \
GLOG_alsologtostderr=1 \
GLOG_stderrthreshold=1 \
GLOG_log_dir=$LOG \
    $TOOLS/caffe train --solver=./solver.prototxt \
    --weights=./models.final.v1/gnet_iter_210000.caffemodel \
    --gpu=3

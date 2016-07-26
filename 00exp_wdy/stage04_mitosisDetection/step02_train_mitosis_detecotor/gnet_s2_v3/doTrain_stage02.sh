#!/usr/bin/env sh
export PYTHONPATH="/home/dywang/dl/caffe/python:/home/dywang/tools:$PATHONPATH"

TOOLS=/home/dywang/dl/caffe/build/tools
LOG=log
MODEL=models
mkdir $LOG
mkdir $MODEL
mkdir images

GLOG_logtostderr=0 \
GLOG_alsologtostderr=1 \
GLOG_stderrthreshold=1 \
GLOG_log_dir=$LOG \
    $TOOLS/caffe train --solver=./solver.prototxt \
    -weights models.final/gnet_iter_45000.caffemodel \
    --gpu=3

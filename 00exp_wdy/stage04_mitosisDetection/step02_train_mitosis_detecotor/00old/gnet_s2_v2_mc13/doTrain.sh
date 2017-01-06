#!/usr/bin/env sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/dywang/dl/deep_visual/LIBS/lib"
export PYTHONPATH="$PYTHONPATH:/home/dywang/dl/deep_visual/LIBS/lib/python2.7/dist-packages"

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
    --gpu=3

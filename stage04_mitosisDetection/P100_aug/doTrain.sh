#!/usr/bin/env bash
module unload anaconda
module load opencv-gui

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

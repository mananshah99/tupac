#!/usr/bin/env sh

TOOLS=/home/dywang/dl/caffe/build/tools

$TOOLS/caffe test -model test.prototxt \
    --weights models/mnet_iter_450000.caffemodel \
    -iterations 100 \
    --gpu=3

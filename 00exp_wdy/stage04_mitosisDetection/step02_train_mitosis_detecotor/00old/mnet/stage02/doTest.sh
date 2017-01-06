#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools

$TOOLS/caffe test -model test.prototxt \
    -weights models/mnet_iter_495000.caffemodel \
    -iterations 1000 \
    --gpu 3

#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools

$TOOLS/caffe test -model test.prototxt \
    -weights models.final/gnet_iter_110899.caffemodel \
    -iterations 100 \
    --gpu 3

#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools

$TOOLS/caffe test -model test.prototxt \
    -weights models.final/round01_gnet_iter_230000.caffemodel \
    -iterations 100 \
    --gpu 0

#    -weights models.final/round01_gnet_iter_150000.caffemodel \
#    -weights models.final/round01_gnet_iter_90000.caffemodel \

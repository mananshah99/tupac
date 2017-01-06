#!/usr/bin/env sh
TOOLS=/home/dywang/dl/caffe/build/tools

$TOOLS/caffe test -model test.prototxt \
    -weights models.final/gnet_iter_230000.caffemodel \
    -iterations 20 \
    --gpu 0

#     -weights models.final/gnet_iter_230000.caffemodel : acc = 0.861

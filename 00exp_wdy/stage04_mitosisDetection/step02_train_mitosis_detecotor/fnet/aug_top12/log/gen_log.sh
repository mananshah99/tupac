#!/bin/bash

rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160828-084615.76848 > tmpinfo

$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

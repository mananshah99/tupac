rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160803-141430.93563 > tmpinfo

$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

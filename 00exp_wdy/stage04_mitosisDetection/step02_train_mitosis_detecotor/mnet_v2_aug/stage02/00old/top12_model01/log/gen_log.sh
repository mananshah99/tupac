rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160826-131022.56688 > tmpinfo

$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

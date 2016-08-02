rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160801-171909.125500 > tmpinfo

$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

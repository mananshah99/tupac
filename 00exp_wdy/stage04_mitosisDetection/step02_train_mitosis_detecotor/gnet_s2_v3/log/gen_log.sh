rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160728-125528.118397 > tmpinfo

$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160720-175505.118165 > tmpinfo

$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160714-150741.58699 > tmpinfo

$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

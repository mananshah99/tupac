rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160822-102253.73309 > tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160822-141034.3240 >> tmpinfo
$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

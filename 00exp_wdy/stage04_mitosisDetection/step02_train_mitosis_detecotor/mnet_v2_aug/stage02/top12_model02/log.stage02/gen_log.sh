rm -rf tmpinfo
cat caffe.deepath-01.dywang.log.INFO.20160828-083659.73602 > tmpinfo

$CAFFE_ROOT_O/tools/extra/parse_log.sh tmpinfo
gnuplot casia.gnuplot

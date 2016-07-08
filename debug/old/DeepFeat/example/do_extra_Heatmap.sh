#!/bin/bash

curdir=`pwd`
root='.'

cd ..
inifile="$root/conf.ini"

inputFolder="$curdir"
inputList="$curdir/wsi.lst"

modelLevel=0
outputDir="$root/results"
cmd="python ./tools/extract_wsi.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 2 \
    --mask_image_level 5 \
    --augmentation 1 \
    --device_ids 1 \
    --gpu"

echo $cmd
exec $cmd

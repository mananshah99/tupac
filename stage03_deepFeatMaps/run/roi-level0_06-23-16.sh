#!/bin/bash

curdir=`pwd`
root='/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run'

cd ..
inifile="$root/../caffe/roi-level0_06-23-16/conf.ini"

inputFolder="$curdir"
inputList="$curdir/wsi.lst"

modelLevel=0
outputDir="$root/../results/roi-level0_06-23-16"
cmd="python ./tools/extract_wsi_tupac.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 2 \
    --mask_image_level 2 \
    --augmentation 1 \
    --device_ids 2 \
    --gpu"

echo $cmd
exec $cmd

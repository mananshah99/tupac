#!/bin/bash

curdir=`pwd`
root='/data/dywang/Database/Proliferation/libs/DeepFeat_manan/example'

cd ..
inifile="$root/conf.ini"

inputFolder="$curdir"
inputList="$curdir/wsi_old2.lst"

modelLevel=0
outputDir="$root/results"
cmd="python ./tools/extract_wsi_tupac.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 2 \
    --mask_image_level 2 \
    --augmentation 1 \
    --device_ids 2 \
    --gpu"

echo $cmd
exec $cmd

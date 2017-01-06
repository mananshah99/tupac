#!/bin/bash

curdir=`pwd`
root='/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run'

cd ..
inifile="$root/../caffe/roi-level1_06-24-16/conf.ini"

inputFolder="$curdir"
inputList="$curdir/wsi-test-bot.lst"

modelLevel=1
outputDir="$root/../results/roi-level1-test_07-11-16"
cmd="python ./tools/extract_wsi_tupac.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 2 \
    --mask_image_level 2 \
    --augmentation 1 \
    --device_ids 3 \
    --gpu"

echo $cmd
exec $cmd

#!/bin/bash

curdir=`pwd`
root='/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run'

cd ..
inifile="$root/../caffe/mitosis_06-21-16/conf.ini"

inputFolder="$curdir"
inputList="$curdir/mitosis.lst"

modelLevel=1
outputDir="$root/../results/mitosis_06-27-16"
cmd="python ./tools/extract_hp_image_tupac.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 2 \
    --augmentation 1 \
    --device_ids 2 \
    --step_size 4 \
    --gpu"

echo $cmd
exec $cmd

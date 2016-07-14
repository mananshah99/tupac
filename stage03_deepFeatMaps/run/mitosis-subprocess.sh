#!/bin/bash

curdir=`pwd`
root='/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run'

cd ..
inifile="$root/../caffe/mitosis_06-29-16/conf.ini"

inputFolder="/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run"
inputList="/data/dywang/Database/Proliferation/libs/METHOD1/tmp.lst"

modelLevel=0
outputDir="$root/../results/mitosis-full_06-29-16"
cmd="python $root/../tools/extract_hp_image_tupac.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 2 \
    --augmentation 1 \
    --device_ids 1 \
    --step_size 4 \
    --gpu"

echo $cmd
exec $cmd

#!/bin/bash

curdir=`pwd`
root='/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run'

cd ..
inifile="$root/../caffe/mitosis_07-12-16/conf.ini"

inputFolder="$curdir"
inputList="$curdir/mitosis-train.lst"

# we're using extract_hp_image_mitosis.py SOLELY for the naming convention in the training images

modelLevel=0
outputDir="$root/../results/mitosis-train-stage2_07-12-16"
cmd="python ./tools/extract_hp_image_mitosis.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --window_size 100 \
    --heatmap_level 2 \
    --augmentation 1 \
    --device_ids 2 \
    --step_size 4 \
    --gpu"

echo $cmd
exec $cmd

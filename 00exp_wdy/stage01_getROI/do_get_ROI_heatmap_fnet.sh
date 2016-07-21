#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'
inifile="$curdir/step01_train_models/LEVEL00/models-fnet-fc-manan/conf.ini"

inputFolder="$root/data/TrainingData"
inputList="$curdir/tmp.lst"

modelLevel=0
outputDir="$curdir/tmp"

mkdir -p $outputDir
cmd="python ./tools/extract_wsi.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 3 \
    --mask_image_level 3 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 100 \
    --step_size 100 \
    --batch_size 256 \
    --group_size 1024 \
    --level_ratio 4 \
    --gpu"

echo $cmd
exec $cmd

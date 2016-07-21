#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'
inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/fnet/conf.ini"

inputFolder="$root/data/mitoses/mitoses_image_data"
#inputList="$root/data/mitoses/image_withmask.lst"
inputList="$curdir/tmp.lst"

modelLevel=0
outputDir="$curdir/tmp"
mkdir -p $outputDir

cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 2 \
    --window_size 100 \
    --step_size 10 \
    --batch_size 256 \
    --group_size 256 \
    --gpu"

echo $cmd
exec $cmd

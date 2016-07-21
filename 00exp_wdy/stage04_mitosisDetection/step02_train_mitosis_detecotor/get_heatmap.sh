#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat
root='/home/dywang/Proliferation'

inifile="$curdir/gnet_s2_v3/conf.ini"
inputFolder="/home/dywang/Proliferation/data/mitoses"
inputList="/home/dywang/Proliferation/data/mitoses/all_image_withmask_mc13_tr.lst"

outputDir="$curdir/heatmap_mc13_tr"
modelLevel=0
mkdir -p $outputDir

program_mode="LOCAL"
cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 100 \
    --step_size 4 \
    --batch_size 256 \
    --group_size 1024 \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
exec $cmd

#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor'

#inifile="$root/mitko/model01/conf_full.ini"
inifile="$root/mitko/model01/conf.ini"

inputFolder="$curdir/wsi"
inputList="$curdir/wsi.lst"
outputDir="$curdir/wsi_cnn"

modelLevel=0
rm -rf $outputDir
mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_wsi_par.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 2 \
    --mask_image_level 2 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 63 \
    --step_size 1000 \
    --batch_size 16 \
    --group_size 256 \
    --level_ratio 4 \
    --overwrite \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
exec $cmd

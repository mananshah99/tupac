#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor'
inifile="$root/mitko/model01/conf_full.ini"
#inifile="$root/mitko/model01/conf.ini"

inputFolder="$curdir/imgs_norm"
inputList="$curdir/test_nm.lst"
outputDir="$curdir/result_mitko_m1_fcnn"

modelLevel=0
mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 1000 \
    --step_size 500 \
    --batch_size 256 \
    --group_size 1024 \
    --overwrite \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
exec $cmd

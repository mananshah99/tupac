#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'

model_folder='mitko/model01'
#model_folder='mitko/model02'

inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/$model_folder/conf.ini"

inputFolder="$root/data/mitoses"
inputList="$root/data/mitoses/all_image_cn1_a13_te.lst"

outputDir="$curdir/$model_folder"

mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_full.py caffe ${inifile} prob ${inputFolder} ${outputDir} ${inputList} \
    --pad 31 \
    --index 1 \
    --shift_step 0 \
    --shift_range 16\
    --device_ids $1 \
    --gpu \
    "
echo $cmd
exec $cmd

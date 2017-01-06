#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'

model_folder='mnet_v2_aug/stage02/top12_model01'
#model_folder='mnet_v2_aug/stage02/top12_model02'
#model_folder='mnet_v2_aug/stage02/top12_model03'
conf_name='conf_2000K'

inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/$model_folder/$conf_name.ini"
inputFolder="$root/data/mitoses"

inputList="$root/data/mitoses/all_image_cn3_a13_te.lst"
outputDir="$curdir/${model_folder}_${conf_name}"

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

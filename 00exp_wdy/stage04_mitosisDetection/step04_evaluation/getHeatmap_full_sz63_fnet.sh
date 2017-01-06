#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'

model_folder='fnet/aug_top12'

inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/$model_folder/conf_$1.ini"
inputFolder="$root/data/mitoses"

inputList="$root/data/mitoses/all_image_cn3_a13_te.lst"
outputDir="$curdir/${model_folder}_${1}"

mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_full.py caffe ${inifile} prob ${inputFolder} ${outputDir} ${inputList} \
    --pad 31 \
    --index 1 \
    --shift 4 \
    --shift_step 4\
    --device_ids 3 \
    --gpu \
    "
echo $cmd
exec $cmd

#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'

inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/mnet_v2/stage02_noaug_top12/conf.ini"
#inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/mnet_v2_aug/stage01/conf.ini"

inputFolder="$root/data/mitoses"
inputList="$root/data/mitoses/all_image_cn3_1.lst"

outputDir="$curdir/tmp"

#outputDir="$curdir/mnet_v2_aug_stage01"

mkdir -p $outputDir
program_mode="LOCAL"

cmd="python ./tools/extract_full.py caffe ${inifile} prob ${inputFolder} ${outputDir} ${inputList} \
    --pad 31 \
    --index 1 \
    --shift 4 \
    --shift_step 4\
    "

echo $cmd
exec $cmd

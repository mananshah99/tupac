#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'

#model_folder='mnet_v2/stage02_noaug_all'
#model_folder='mnet_v2/stage02_noaug_top12'
#model_folder='mnet_v2/stage02_badaug_all'
#model_folder='mitko/model01'
#model_folder='mitko/model02'
#model_folder='mnet_v2_aug/stage01'
#model_folder='mnet_v2_aug/stage02/top12_model01'
#model_folder='mnet_v2_aug/stage02/top12_model02'
model_folder='mnet_v2_aug/stage02/top12_model03'
#model_folder='fnet/aug_top12'
#model_folder='gnet'

inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/$model_folder/conf.ini"
inputFolder="$root/data/mitoses"
inputList="$root/data/mitoses/all_image_cn3_1.lst"

outputDir="$curdir/full/$model_folder"

mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_full.py caffe ${inifile} prob ${inputFolder} ${outputDir} ${inputList} \
    --pad 32 \
    --index 1 \
    --shift 0 \
    --shift_step 4\
    --device_ids 1 \
    --gpu \
    "
echo $cmd
exec $cmd

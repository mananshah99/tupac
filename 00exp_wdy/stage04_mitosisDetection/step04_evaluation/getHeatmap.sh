#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'

#model_folder='gnet'
model_folder='mnet_v2_aug/stage02/top12_model05'


#######
inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/$model_folder/conf.ini"
inputFolder="$root/data/mitoses"
inputList="$root/data/mitoses/all_image_cn3_withmask_1.lst"

outputDir="$curdir/$model_folder"

mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_image.py caffe ${inifile} prob ${inputFolder} ${inputList} 0 ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids $1 \
    --window_size 64 \
    --step_size 4 \
    --batch_size 256 \
    --group_size 4096 \
    --program_mode ${program_mode} \
    --cluster_stage LIST \
    --parallel 0 \
    "

echo $cmd
exec $cmd

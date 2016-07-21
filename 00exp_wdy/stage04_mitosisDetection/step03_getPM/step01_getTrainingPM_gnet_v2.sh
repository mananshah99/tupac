#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'
inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/gnet_s2_v2/conf.ini"

inputFolder="$root/data/mitoses/mitoses_image_data"
#inputList="$root/data/mitoses/image_withmask.lst"
#outputDir="$curdir/stage1_gnet"

inputList="$curdir/tmp.lst"
outputDir="$curdir/tmp2"

modelLevel=0
mkdir -p $outputDir

program_mode="LOCAL"

cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 100 \
    --step_size 8 \
    --batch_size 256 \
    --group_size 1024 \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
exec $cmd

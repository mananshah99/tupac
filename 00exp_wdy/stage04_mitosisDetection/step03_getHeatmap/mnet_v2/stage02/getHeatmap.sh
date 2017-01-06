#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'
#inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/mnet_v2/stage01/conf.ini"
#inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/mnet_v2/stage02_m4/conf.ini"

inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/mnet_v2/stage02_m7/conf.ini"

#inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/mnet_v2/stage02_m8/conf.ini"

#inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/fnet/conf.ini"

#inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/mnet_v2_aug/stage01/conf.ini"

inputFolder="$root/data/mitoses"
inputList="$root/data/mitoses/all_image_cn3_withmask_1.lst"


modelLevel=0

outputDir="$curdir/mnet_stage02_m7_200K"

#outputDir="$curdir/mnet_stage02_m8_500K"

#outputDir="$curdir/fnet"

#outputDir="$curdir/mnet_v2_aug_stage01"

mkdir -p $outputDir
program_mode="LOCAL"

cmd="python ./tools/extract_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
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

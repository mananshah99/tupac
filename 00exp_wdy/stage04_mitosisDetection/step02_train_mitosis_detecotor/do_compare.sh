#!/bin/bash

curdir=`pwd`
cd /home/dywang/dl/DeepFeat
root='/home/dywang/Proliferation'

inifile="$curdir/mitko/model01/conf.ini"
inputFolder="$curdir/imgs_norm"
inputList="$curdir/test_nm.lst"
outputDir="$curdir/result_mitko_m1"

modelLevel=0
mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 63 \
    --step_size 4 \
    --batch_size 256 \
    --group_size 1024 \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
#exec $cmd

###############################

inifile="$curdir/mitko/model02/conf.ini"
inputFolder="$curdir/imgs"
inputList="$curdir/test.lst"
outputDir="$curdir/result_mitko_m2"

modelLevel=0
mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 63 \
    --step_size 4 \
    --batch_size 256 \
    --group_size 1024 \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
#exec $cmd

###############################

inifile="$curdir/gnet_s2_v2/conf.ini"
inputFolder="$curdir/imgs"
inputList="$curdir/test.lst"
outputDir="$curdir/result_gnet_full_v2"

modelLevel=0
mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 100 \
    --step_size 4 \
    --batch_size 256 \
    --group_size 1024 \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
#exec $cmd

###############################

inifile="$curdir/gnet_s2_v2_mc13/conf.ini"
inputFolder="$curdir/imgs"
inputList="$curdir/test.lst"
outputDir="$curdir/result_gnet_full_v2_mc13"

modelLevel=0
mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 100 \
    --step_size 4 \
    --batch_size 256 \
    --group_size 1024 \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
#exec $cmd

###############################

inifile="$curdir/gnet_s2_v3/conf.ini"
inputFolder="$curdir/imgs_norm"
inputList="$curdir/test_nm.lst"
outputDir="$curdir/result_gnet_v3"

modelLevel=0
mkdir -p $outputDir
program_mode="LOCAL"
cmd="python ./tools/extract_hp_image.py caffe ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --heatmap_level 0 \
    --mask_image_level 0 \
    --augmentation 1 \
    --device_ids 3 \
    --window_size 100 \
    --step_size 4 \
    --batch_size 256 \
    --group_size 1024 \
    --program_mode ${program_mode} \
    --gpu"

echo $cmd
#exec $cmd

###############################

inifile="$curdir/gnet_s2_v4/conf.ini"
inputFolder="$curdir/imgs_norm"
inputList="$curdir/test_nm.lst"
outputDir="$curdir/result_gnet_v4"

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

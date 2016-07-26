#!/bin/bash


# /data/dywang/Database/Proliferation/evaluation/00models/googlenet/
curdir=`pwd`
root='/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/run'

cd ..

inifile="$root/../caffe/mitosis-wsi/conf.ini"

inputFolder="$curdir"
inputList="$curdir/wsi-full-bot.lst" #this is the wsi and the TISSUE MASK

modelLevel=0
outputDir="$root/../results/mitosis-wsi-tst"

root=''
cmd="python ./tools/extract_wsi_full_tupac.py caffe-no-resize ${inifile} prob ${inputFolder} ${inputList} ${modelLevel} ${outputDir} \
    --window_size 1000 \
    --heatmap_level 2 \
    --mask_image_level 2 \
    --augmentation 1 \
    --device_ids 1 \
    --step_size 500 \
    --batch_size 32
    --gpu"

echo $cmd
exec $cmd

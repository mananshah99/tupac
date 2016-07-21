#! /bin/sh
#
# step01_getTrainingPM_gnet_full.sh
# Copyright (C) 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.
#

curdir=`pwd`
cd /home/dywang/dl/DeepFeat

root='/home/dywang/Proliferation'
inifile="$root/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/gnet/conf.ini"

inputFolder="$root/data/mitoses/mitoses_image_data"
#inputList="$root/data/mitoses/image_withmask.lst"
inputList="$curdir/tmp.lst"

modelLevel=0
outputDir="$curdir/stage1_gnet_full"
mkdir -p $outputDir

program_mode="LOCAL"

cmd="python ./tools/extract.py caffe ${inifile} prob ${inputFolder}  ${outputDir} ${inputList} \
    --device_ids 3 \
    --gpu"

echo $cmd
exec $cmd

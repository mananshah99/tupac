#!/bin/bash

curdir=`pwd`

cmd="python /home/dywang/WWW/deepzoom/visual_v2.py -l 10.35.73.88 -p 5018 -n heatmap \
    -r ${curdir}/result_wsi_fcnn/heatmap \
    --debug \
    --heatmap_threshold 0.2 \
    --overlap_ratio 0.1 \
    --color_type 2 \
    /home/dywang/Proliferation/data/TrainingData/small_images-level-3"

echo $cmd
exec $cmd
